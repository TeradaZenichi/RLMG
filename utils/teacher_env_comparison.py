# replay_batt_only.py
# Replays MILP battery power in the Gymnasium env; env decides curtailment.
# Comments in English (compact).

from datetime import datetime
import json
import numpy as np
import pandas as pd

from opt import Teacher
from environment import EnergyEnv  # your env class (Δ, β), no overrides

# Adjust project root for imports
import sys
import os
target_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(target_path)

START_TIME = "2006-12-17 00:00:00"
START_SOC  = 0.5  # fraction of Emax (same used in Teacher.build)

def hour_key(ts: pd.Timestamp) -> str:
    return f"{int(ts.hour):02d}:00"

if __name__ == "__main__":
    # --- Load params and build MILP (Teacher) ---
    params = json.load(open("data/parameters.json", "r"))
    start_time = datetime.strptime(START_TIME, "%Y-%m-%d %H:%M:%S")

    teacher = Teacher.Teacher(params, None, None)
    teacher.build(start_time, START_SOC)
    teacher.solve()
    df = teacher.results2dataframe()          # index = timestamps
    dt_h = params["time"]["timestep"] / 60.0

    # --- Compute MILP cost (as in the objective) ---
    price_map = params["costs"]["EDS"]                   # {"HH:00": price}
    cload_shed = float(params["costs"]["load_shedding"]) # penalty per kWh shed
    df["tariff"] = [ price_map[hour_key(t)] for t in df.index ]

    df["E_import_kWh"] = df["Peds_in"] * dt_h
    cost_grid_milp = (df["E_import_kWh"] * df["tariff"]).sum()

    # MILP may shed load: count it (env won't shed)
    df["Shed_kw"] = (df["load_required"] - df["load"]).clip(lower=0.0)
    df["E_shed_kWh"] = df["Shed_kw"] * dt_h
    cost_shed_milp = (df["E_shed_kWh"] * cload_shed).sum()

    cost_total_milp = cost_grid_milp + cost_shed_milp
    print(f"[MILP] grid={cost_grid_milp:.6f}  shed={cost_shed_milp:.6f}  total={cost_total_milp:.6f}")

    # --- Prepare env params (same start_time / horizon) ---
    params_env = json.loads(json.dumps(params))  # deep copy
    params_env["time"]["start_time"]    = START_TIME
    params_env["time"]["horizon_hours"] = len(df) * dt_h
    # keep curtailment penalty = 0 to mirror MILP default (adjust if needed)
    params_env.setdefault("costs", {}).setdefault("c_pv_curt_per_kwh", 0.0)

    # --- Create env (no PV/LOAD overrides) ---
    env = EnergyEnv(params_env)
    obs, info = env.reset()

    # --- Replay: feed *only* battery power series from MILP ---
    total_env_cost = 0.0
    total_env_grid_cost = 0.0

    # Battery power sequence from MILP.
    # It might be either normalized ([-1,1]) or in kW already; detect and scale.
    pb_raw = df["Pbess"].astype(float).copy()
    pmax = float(params["BESS"]["Pmax"])

    # Auto-detect normalization: if max |pb| <= ~1, treat as normalized.
    # A small margin (1.2) is used to be robust to numerical noise.
    is_normalized = (np.nanmax(np.abs(pb_raw.values)) <= 1.2)
    if is_normalized:
        print("[Replay] Detected normalized Pbess in [-1,1]. Rescaling by Pmax.")
        pb_kw = (pb_raw * pmax).clip(lower=-pmax, upper=pmax)
    else:
        print("[Replay] Detected Pbess in kW. Clipping to [-Pmax, Pmax].")
        pb_kw = pb_raw.clip(lower=-pmax, upper=pmax)

    for tstamp, pb in pb_kw.items():
        action = np.array([pb], dtype=np.float32)  # env expects kW action
        obs, reward, terminated, truncated, step_info = env.step(action)
        row = step_info["row"]

        # accumulate env cost (identical to reward definition)
        total_env_cost += float(row["cost_total"])

        # accumulate grid component for visibility
        price = float(step_info.get("tariff", step_info.get("price_cperkwh", 0.0)))
        total_env_grid_cost += price * float(row["P_grid_in_kw"]) * dt_h

        if terminated or truncated:
            break

    print(f"[ENV ] grid={total_env_grid_cost:.6f}  (no shed)  total={total_env_cost:.6f}")
    print(f"[Δ] ENV − MILP = {total_env_cost - cost_total_milp:.6f}")
