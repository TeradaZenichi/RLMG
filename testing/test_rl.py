# test/test_base.py
# -*- coding: utf-8 -*-
from datetime import datetime
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch

from opt import Teacher
from environment import EnergyEnv
from utils.plot_results import plot_solution_from_df


def hour_key(ts: pd.Timestamp) -> str:
    return f"{int(ts.hour):02d}:00"


def make_history_X(states, cfg, *, force_flat: bool = True):
    S = np.asarray(states, dtype=np.float32)
    if S.ndim != 2 or S.size == 0:
        return S

    N, F = S.shape
    hist_cfg = (cfg.get("obs", {}) or {}).get("history", {}) if isinstance(cfg, dict) else {}
    enabled  = bool(hist_cfg.get("enabled", False))
    if not enabled:
        return S  # comportamento antigo

    K       = int(hist_cfg.get("window", 12))
    stride  = int(hist_cfg.get("stride", 1))
    mode    = str(hist_cfg.get("mode", "flat")).lower()
    padding = str(hist_cfg.get("padding", "edge")).lower()

    K = max(1, K)
    stride = max(1, stride)

    # padding à esquerda para formar a primeira janela
    pad_left = max(0, K - 1)
    if pad_left > 0:
        if padding == "zero":
            pad_block = np.zeros((pad_left, F), dtype=np.float32)
        elif padding == "circular" and N >= pad_left:
            pad_block = S[-pad_left:, :]
        else:  # "edge" (default) ou fallback
            pad_block = np.repeat(S[:1, :], repeats=pad_left, axis=0)
        S_pad = np.concatenate([pad_block, S], axis=0)  # [N+K-1, F]
    else:
        S_pad = S  # K=1 não precisa de padding

    X_list = []
    for i in range(0, N, stride):
        w = S_pad[i:i+K, :]
        if w.shape[0] < K:
            break
        if (mode == "sequence") and (not force_flat):
            X_list.append(w)                 # [K, F]
        else:
            X_list.append(w.reshape(K * F))  # [K*F]

    if not X_list:
        # fallback caso N << K
        return S if force_flat else S.reshape(S.shape[0], 1, F)

    X = np.stack(X_list, axis=0)
    return X


def test_policy(
    start_time: str,
    start_soc: float,
    model_arch,          # ex.: models.model2.model.MLPmodel
    hp_cls,              # ex.: models.model2.model.Hyperparameters
    config_path: str,    # ex.: "models/model2/config.json"
    name: str,           # usado como pasta em saves/<name>
    params_path: str = "data/parameters.json",
    save_plots: bool = True,
    save_tables: bool = True,
    force_flat: bool = True,  # mantém compatibilidade com MLP/input_dim do checkpoint
):
    # --- paths / cfg ---
    save_dir  = Path(f"saves/{name.replace(' ', '_')}")
    ckpt_path = save_dir / "best.pt"
    hp_path   = save_dir / "hp.json"

    params = json.load(open(params_path, "r"))
    dt_h = params["time"]["timestep"] / 60.0
    Pmax = float(params["BESS"]["Pmax"])
    Emax = float(params["BESS"]["Emax"])
    c_shed = float(params["costs"]["load_shedding"])

    # --- MILP (Teacher) ---
    start_dt = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
    teacher = Teacher.Teacher(params, None, None)
    teacher.build(start_dt, start_soc)
    teacher.solve()
    df_milp = teacher.results2dataframe().sort_index()
    df_milp["tariff"] = [ params["costs"]["EDS"][hour_key(t)] for t in df_milp.index ]

    # --- Política (carrega cfg/ckpt) ---
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    # preferir a cfg salva no checkpoint (garante mesma config de treino/histórico)
    if hp_path.exists():
        cfg_fallback = json.load(open(hp_path, "r"))
    else:
        cfg_fallback = json.load(open(config_path, "r"))
    cfg_ckpt = ckpt.get("cfg", cfg_fallback)

    hp = hp_cls(cfg_ckpt)
    input_dim = int(ckpt["input_dim"])  # dim salvo no treino IL

    net = model_arch(hp, input_dim=input_dim)
    net.load_state_dict(ckpt["model_state"])
    net.eval()

    # --- monta X com histórico (padronizado com o train) ---
    states = teacher.normalized_states                   # [N,F] já normalizados
    X = make_history_X(states, cfg_ckpt, force_flat=force_flat)  # [M,F] ou [M,K*F] ou [M,K,F]
    if X.ndim == 3:                                      # garante compatibilidade com MLP
        X = X.reshape(X.shape[0], -1)
    x = torch.tensor(X, dtype=torch.float32)

    # --- inferência batelada ---
    with torch.no_grad():
        a_norm = net(x)["action"].cpu().numpy()          # [-1,1]
    pb_policy_kw = (a_norm * Pmax).clip(-Pmax, Pmax)

    # --- Ambiente (mesma janela) ---
    params_env = json.loads(json.dumps(params))
    env = EnergyEnv(params_env,
                    start_time=start_time,
                    soc_ini=start_soc,
                    horizon_hours=len(df_milp)*dt_h,
                    timestep=params_env["time"]["timestep"])
    env.reset(options={"E_init_kwh": start_soc * Emax})   # garante SoC inicial

    # --- Rollout da política: df para plot e custos ---
    rows = []
    for (tstamp, _row_m), pb in zip(df_milp.iterrows(), pb_policy_kw.tolist()):
        _, _, terminated, truncated, info = env.step(np.array([pb], dtype=np.float32))
        r = info.get("row", {})
        rows.append({
            "timestamp": tstamp,
            "load": float(r.get("Load_served_kw", 0.0)),
            "pv":   float(r.get("PV_used_kw", 0.0)),
            "Peds": float(r.get("P_grid_in_kw", 0.0) - r.get("P_grid_out_kw", 0.0)),
            "Ebess": float(r.get("E_kwh", start_soc * Emax)),
            "Pbess": float(r.get("P_bess_kw", pb)),
            "load_required": float(r.get("Load_kw", r.get("Load_served_kw", 0.0))),
            "pv_available":  float(r.get("PV_kw",  r.get("PV_used_kw", 0.0))),
            "P_grid_in_kw":  float(r.get("P_grid_in_kw", 0.0)),
            "P_grid_out_kw": float(r.get("P_grid_out_kw", 0.0)),
            "Curtailment_kw": float(r.get("Curtailment_kw", 0.0)),
            "Shedding_kw":    float(r.get("Shedding_kw", 0.0)),
            "tariff": float(info.get("tariff", 0.0)),
            "cost_total": float(r.get("cost_total", 0.0)),
        })
        if terminated or truncated:
            break

    df_pol = pd.DataFrame(rows).set_index("timestamp").sort_index()

    # --- Plots em PDF ---
    save_dir.mkdir(parents=True, exist_ok=True)
    if save_plots:
        plot_solution_from_df(
            df_milp, e_nom_kwh=Emax,
            title="MILP (Teacher) — Powers & SoC",
            save_path=str(save_dir / "plot_milp.pdf"), show=False
        )
        plot_solution_from_df(
            df_pol,  e_nom_kwh=Emax,
            title=f"Policy ({name}) — Powers & SoC",
            save_path=str(save_dir / "plot_policy.pdf"), show=False
        )

    # --- Salvar DFs ---
    if save_tables:
        df_milp.to_csv(save_dir / "ops_milp.csv")
        df_pol.to_csv(save_dir / "ops_policy.csv")
        try:
            df_milp.to_parquet(save_dir / "ops_milp.parquet", index=True)
            df_pol.to_parquet(save_dir / "ops_policy.parquet", index=True)
        except Exception:
            pass

    # --- Custos finais (PRINTS ORIGINAIS MANTIDOS) ---
    cost_grid_milp = float((df_milp.get("Peds_in", 0.0) * df_milp["tariff"] * dt_h).sum()) if "Peds_in" in df_milp else 0.0
    shed_kw = (df_milp["load_required"] - df_milp["load"]).clip(lower=0.0) if {"load_required","load"} <= set(df_milp.columns) else pd.Series(0.0, index=df_milp.index)
    cost_shed_milp = float((shed_kw * c_shed * dt_h).sum())
    cost_total_milp = cost_grid_milp + cost_shed_milp

    cost_total_policy = float(df_pol["cost_total"].sum()) if "cost_total" in df_pol else np.nan
    cost_grid_policy  = float((df_pol["P_grid_in_kw"] * df_pol["tariff"] * dt_h).sum()) if {"P_grid_in_kw","tariff"} <= set(df_pol.columns) else np.nan

    print("\n=== Cost Summary ===")
    print(f"[MILP ] grid={cost_grid_milp:.6f}  shed={cost_shed_milp:.6f}  total={cost_total_milp:.6f}")
    print(f"[POL  ] grid={cost_grid_policy:.6f}  total={cost_total_policy:.6f}")
    print(f"[Δ    ] Policy.total - MILP.total = {cost_total_policy - cost_total_milp:.6f}")

    print(f"\nOK: plots em PDF e DFs salvos em {save_dir}/")

    # opcional: retorno programático
    return {
        "df_milp": df_milp,
        "df_policy": df_pol,
        "save_dir": save_dir
    }
