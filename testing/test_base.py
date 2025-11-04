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
    """Return the hour key as 'HH:00' for tariff lookup."""
    return f"{int(ts.hour):02d}:00"


# --- Build X with temporal history (stride=1) when cfg['obs']['history'] is enabled ---
def make_history_X(states, cfg):
    """
    Build a flattened temporal window for MLP inference.

    Args:
        states: array-like [N, F], already normalized
        cfg: dict that may contain cfg['obs']['history'] with:
             - enabled: bool
             - window: int (K)
             - padding: 'edge' | 'zero' | 'circular'

    Returns:
        X with shape [N, F] if history disabled, or [N, K*F] if enabled.
        Uses stride=1 for test time (one step per env.step).
    """
    S = np.asarray(states, dtype=np.float32)
    N, F = S.shape[0], S.shape[1]
    hist_cfg = (cfg.get("obs", {}) or {}).get("history", {}) if isinstance(cfg, dict) else {}
    enabled = bool(hist_cfg.get("enabled", False))
    if not enabled:
        return S  # legacy behavior (no history)

    K = int(hist_cfg.get("window", 12))
    padding = str(hist_cfg.get("padding", "edge")).lower()

    pad_left = max(0, K - 1)
    if padding == "zero":
        pad_block = np.zeros((pad_left, F), dtype=np.float32)
    elif padding == "circular" and N >= pad_left:
        pad_block = S[-pad_left:, :]
    else:  # "edge" (default)
        pad_block = np.repeat(S[:1, :], repeats=pad_left, axis=0)

    S_pad = np.concatenate([pad_block, S], axis=0)  # [N+K-1, F]

    X_list = []
    for i in range(N):  # stride=1 for evaluation
        w = S_pad[i:i + K, :]
        if w.shape[0] < K:
            break
        X_list.append(w.reshape(K * F))  # flatten for MLP
    X = np.stack(X_list, axis=0)
    return X


def test(
    start_time: str,
    start_soc: float,
    model_arch,          # e.g., models.model2.model.MLPmodel
    hp_cls,              # e.g., models.model2.model.Hyperparameters
    config_path: str,    # e.g., "models/model2/config.json"
    name: str,           # used as folder under saves/<name>
    params_path: str = "data/parameters.json",
    save_plots: bool = True,
    save_tables: bool = True,
):
    """
    Evaluate the IL policy (checkpoint best.pt) on the same time window used by MILP (Teacher).
    Keeps original PRINTs at the end.
    """
    # --- paths / cfg ---
    save_dir = Path(f"saves/{name.replace(' ', '_')}")
    ckpt_path = save_dir / "best_il.pt"
    hp_path = save_dir / "hp.json"

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
    df_milp["tariff"] = [params["costs"]["EDS"][hour_key(t)] for t in df_milp.index]

    # --- Policy (load hp.json + checkpoint) ---
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    # Prefer cfg saved in checkpoint (ensures exact training config)
    cfg_ckpt = ckpt.get(
        "cfg",
        json.load(open(hp_path, "r")) if hp_path.exists() else json.load(open(config_path, "r"))
    )
    hp = hp_cls(cfg_ckpt)

    # Input dim saved at training (must match X we build)
    input_dim = int(ckpt["input_dim"])

    net = model_arch(hp, input_dim=input_dim)
    net.load_state_dict(ckpt["model_state"])
    net.eval()

    # --- Build X from Teacher normalized states (with optional history) ---
    states = teacher.normalized_states                   # [N, F] already normalized
    X = make_history_X(states, cfg_ckpt)                 # [N, F] or [N, K*F]
    x = torch.tensor(X, dtype=torch.float32)

    # --- Batched inference (same length as MILP window) ---
    with torch.no_grad():
        # Model outputs normalized action in [-1, 1]
        a_norm = net(x)["action"].cpu().numpy().astype(np.float32)

    # Clamp to valid action space (robustness)
    a_env = np.clip(a_norm, -1.0, 1.0)                   # this is what the env expects
    pb_policy_kw = (a_env * Pmax)                        # for reporting/plots only

    # --- Environment (same time window) ---
    params_env = json.loads(json.dumps(params))
    env = EnergyEnv(
        params_env,
        start_time=start_time,
        soc_ini=start_soc,
        horizon_hours=len(df_milp) * dt_h,
        timestep=params_env["time"]["timestep"],
        # if your env supports a window_size, add it here, e.g. window_size=hp.obs.window
    )
    # No need to pass custom options to force SoC; env uses soc_ini

    # --- Rollout policy: build DF for plots & costs ---
    rows = []
    a_list = a_env.reshape(-1).tolist()
    pb_list = pb_policy_kw.reshape(-1).tolist()

    for (tstamp, _row_m), a_i, pb_i in zip(df_milp.iterrows(), a_list, pb_list):
        # IMPORTANT: pass normalized action to the environment ([-1, 1])
        _, _, terminated, truncated, info = env.step(np.array([a_i], dtype=np.float32))

        r = info.get("row", {})
        rows.append({
            "timestamp": tstamp,
            "load": float(r.get("Load_served_kw", 0.0)),
            "pv":   float(r.get("PV_used_kw", 0.0)),
            "Peds": float(r.get("P_grid_in_kw", 0.0) - r.get("P_grid_out_kw", 0.0)),
            "Ebess": float(r.get("E_kwh", start_soc * Emax)),
            # Prefer env log for Pbess; fallback to predicted pb_i (kW)
            "Pbess": float(r.get("P_bess_kw", pb_i)),
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

    # --- PDF plots (saved under model folder) ---
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

    # --- Save DFs ---
    if save_tables:
        df_milp.to_csv(save_dir / "ops_milp.csv")
        df_pol.to_csv(save_dir / "ops_policy.csv")
        try:
            df_milp.to_parquet(save_dir / "ops_milp.parquet", index=True)
            df_pol.to_parquet(save_dir / "ops_policy.parquet", index=True)
        except Exception:
            pass

    # --- Final costs (KEEP original prints) ---
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

    print(f"\nOK: PDF plots and DFs saved in {save_dir}/")

    # Return objects for programmatic use
    return {
        "df_milp": df_milp,
        "df_policy": df_pol,
        "save_dir": save_dir
    }
