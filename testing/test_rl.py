# testing/test_rl.py
# -*- coding: utf-8 -*-
from pathlib import Path
from datetime import datetime
import importlib
import json
import numpy as np
import pandas as pd
import torch

from environment import EnergyEnv
from utils.plot_results import plot_solution_from_df


def hour_key(ts: pd.Timestamp) -> str:
    return f"{int(ts.hour):02d}:00"


def _import_actor_from_config(config_path: str):
    """
    Resolve o módulo do ator a partir do caminho do config.
    Ex.: "models/td3mlp/model.json" -> importa "models.td3mlp.model".
    """
    cfg_path = Path(config_path)
    # models/td3mlp/model.json -> models.td3mlp.model
    model_mod = ".".join(cfg_path.with_suffix("").parts).replace(".json", "")
    if not model_mod.endswith(".model"):
        model_mod = f"{model_mod}.model"
    mod = importlib.import_module(model_mod)
    if not hasattr(mod, "DeterministicTanhActor"):
        raise AttributeError(f"DeterministicTanhActor não encontrado em {model_mod}")
    if not hasattr(mod, "Hyperparameters"):
        raise AttributeError(f"Hyperparameters não encontrado em {model_mod}")
    return mod


def test(
    start_time: str,
    start_soc: float,
    hp_cls,                 # Hyperparameters class (não é usado p/ descobrir o módulo)
    config_path: str,       # ex.: "models/td3mlp/model.json"
    name: str,              # pasta: saves/<name>
    params_path: str = "data/parameters.json",
    ckpt_name: str = "best_rl.pt",
    save_plots: bool = True,
    save_tables: bool = True,
):
    # ---------- paths / cfg ----------
    save_dir  = Path(f"saves/{name.replace(' ', '_')}")
    ckpt_path = save_dir / ckpt_name
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint não encontrado: {ckpt_path}")

    params = json.load(open(params_path, "r"))
    dt_h   = float(params["time"]["timestep"]) / 60.0
    Pmax   = float(params["BESS"]["Pmax"])
    Emax   = float(params["BESS"]["Emax"])
    c_shed = float(params["costs"]["load_shedding"])

    # ---------- MILP (Teacher) ----------
    start_dt = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
    from opt import Teacher
    teacher = Teacher.Teacher(params, None, None)
    teacher.build(start_dt, start_soc)
    teacher.solve()
    df_milp = teacher.results2dataframe().sort_index()
    df_milp["tariff"] = [params["costs"]["EDS"][hour_key(t)] for t in df_milp.index]

    # ---------- carrega ckpt TD3 ----------
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    cfg_ckpt = ckpt.get("cfg", json.load(open(config_path, "r")))
    obs_dim = int(ckpt["obs_dim"])
    act_dim = int(ckpt["act_dim"])

    # importa o ator a partir do config_path (mesmo pacote do treino)
    mod = _import_actor_from_config(config_path)
    ActorCls = getattr(mod, "DeterministicTanhActor")
    HPcls    = getattr(mod, "Hyperparameters")
    hp       = HPcls(cfg_ckpt)

    actor = ActorCls(hp, input_dim=obs_dim, action_dim=act_dim)
    actor.load_state_dict(ckpt["actor_state"])
    actor.eval()

    # ---------- Ambiente ----------
    F_BASE = 12
    K = max(1, obs_dim // F_BASE)  # deriva window_size do obs_dim salvo
    env = EnergyEnv(
        json.loads(json.dumps(params)),
        start_time=start_time,
        soc_ini=start_soc,
        horizon_hours=len(df_milp) * dt_h,
        timestep=params["time"]["timestep"],
        window_size=K
    )

    # ---------- Rollout ----------
    rows = []
    with torch.no_grad():
        s, _ = env.reset()
        for tstamp, _ in df_milp.iterrows():
            s_t = torch.as_tensor(s, dtype=torch.float32).unsqueeze(0)  # [1, obs_dim]
            a_norm = actor(s_t)["action"].cpu().numpy().reshape(-1)     # [-1,1]
            s2, _, terminated, truncated, info = env.step(a_norm.astype(np.float32))
            r = info.get("row", {})
            rows.append({
                "timestamp": tstamp,
                "load": float(r.get("Load_served_kw", 0.0)),
                "pv":   float(r.get("PV_used_kw", 0.0)),
                "Peds": float(r.get("P_grid_in_kw", 0.0) - r.get("P_grid_out_kw", 0.0)),
                "Ebess": float(r.get("E_kwh", start_soc * Emax)),
                "Pbess": float(r.get("P_bess_kw", 0.0)),
                "Pbess_set_kw": float(a_norm[0] * Pmax),
                "load_required": float(r.get("Load_kw", r.get("Load_served_kw", 0.0))),
                "pv_available":  float(r.get("PV_kw",  r.get("PV_used_kw", 0.0))),
                "P_grid_in_kw":  float(r.get("P_grid_in_kw", 0.0)),
                "P_grid_out_kw": float(r.get("P_grid_out_kw", 0.0)),
                "Curtailment_kw": float(r.get("Curtailment_kw", 0.0)),
                "Shedding_kw":    float(r.get("Shedding_kw", 0.0)),
                "tariff": float(info.get("tariff", 0.0)),
                "cost_total": float(r.get("cost_total", 0.0)),
            })
            s = s2
            if terminated or truncated:
                break

    df_pol = pd.DataFrame(rows).set_index("timestamp").sort_index()

    # ---------- Plots / Tabelas ----------
    save_dir.mkdir(parents=True, exist_ok=True)
    if save_plots:
        plot_solution_from_df(
            df_milp, e_nom_kwh=Emax,
            title="MILP (Teacher) — Powers & SoC",
            save_path=str(save_dir / "plot_milp.pdf"), show=False
        )
        plot_solution_from_df(
            df_pol,  e_nom_kwh=Emax,
            title=f"TD3 Policy ({name}) — Powers & SoC",
            save_path=str(save_dir / "plot_policy_td3.pdf"), show=False
        )

    if save_tables:
        df_milp.to_csv(save_dir / "ops_milp.csv")
        df_pol.to_csv(save_dir / "ops_policy_td3.csv")
        try:
            df_milp.to_parquet(save_dir / "ops_milp.parquet", index=True)
            df_pol.to_parquet(save_dir / "ops_policy_td3.parquet", index=True)
        except Exception:
            pass

    # ---------- Sumário de custos ----------
    cost_grid_milp = float((df_milp.get("Peds_in", 0.0) * df_milp["tariff"] * dt_h).sum()) if "Peds_in" in df_milp else 0.0
    shed_kw = (df_milp["load_required"] - df_milp["load"]).clip(lower=0.0) if {"load_required","load"} <= set(df_milp.columns) else pd.Series(0.0, index=df_milp.index)
    cost_shed_milp = float((shed_kw * c_shed * dt_h).sum())
    cost_total_milp = cost_grid_milp + cost_shed_milp

    cost_total_policy = float(df_pol["cost_total"].sum()) if "cost_total" in df_pol else np.nan
    cost_grid_policy  = float((df_pol["P_grid_in_kw"] * df_pol["tariff"] * dt_h).sum()) if {"P_grid_in_kw","tariff"} <= set(df_pol.columns) else np.nan

    print("\n=== Cost Summary (TD3) ===")
    print(f"[MILP ] grid={cost_grid_milp:.6f}  shed={cost_shed_milp:.6f}  total={cost_total_milp:.6f}")
    print(f"[TD3  ] grid={cost_grid_policy:.6f}  total={cost_total_policy:.6f}")
    print(f"[Δ    ] TD3.total - MILP.total = {cost_total_policy - cost_total_milp:.6f}")
    print(f"\nOK: arquivos salvos em {save_dir}/")

    return {"df_milp": df_milp, "df_policy": df_pol, "save_dir": save_dir}
