
# eval_day_fixed.py — carregar ator SAC e avaliar um dia (com DataFrame) — versão corrigida
from __future__ import annotations
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from pathlib import Path

from gymnasium.wrappers import TimeLimit
from environment import EnergyEnvSimpleNP
from environment.config import EnergyEnvConfig
from models.model1.model import make_sac_nets


# -----------------------
# Utilidades
# -----------------------

def _iget(d: dict, keys: list[str], default=None):
    """Retorna o primeiro valor presente em d para as chaves; se nada existir, retorna default."""
    for k in keys:
        if k in d:
            return d[k]
    return default


def _safe_get_soc(info_dict: dict, obs_vec):
    """Prefere SoC do info; se não houver, tenta fallback do obs na posição 5."""
    for k in ["soc_frac", "soc", "SoC", "soc_norm"]:
        if k in info_dict:
            return float(info_dict[k])
    try:
        return float(obs_vec[5])
    except Exception:
        return np.nan


# -----------------------
# Redes / checkpoint
# -----------------------

def load_actor(ckpt_path: str):
    """Reconstrói as redes a partir do checkpoint e retorna o ator em modo eval()."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    obs_dim  = int(ckpt["obs_dim"])
    act_dim  = int(ckpt["act_dim"])
    act_low  = np.array(ckpt["act_low"], dtype=np.float32)
    act_high = np.array(ckpt["act_high"], dtype=np.float32)
    net_hp   = ckpt["net_hp"]

    nets = make_sac_nets(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden=net_hp.get("hidden_dims", [256, 256]),
        activation=net_hp.get("activation", "silu"),
        init=net_hp.get("init", "orthogonal"),
        log_std_bounds=(net_hp.get("log_std_min", -5.0), net_hp.get("log_std_max", 2.0)),
        action_low=act_low,
        action_high=act_high,
    )
    actor = nets.actor.to(device)
    actor.load_state_dict(ckpt["actor"])
    actor.eval()
    return actor, device


# -----------------------
# Avaliação de 1 dia
# -----------------------

def evaluate_day(start_date: str,
                 dt_minutes: int = 15,
                 horizon_hours: int = 24,
                 parameters_dir: str = "data",
                 ckpt_path: str = "saves/model1/sac_final.pt",
                 align: str = "exact",
                 save_csv_path: str | None = None):
    """
    Executa política determinística por 'horizon_hours' a partir de 'start_date'.
    Retorna dict com séries e DataFrame 'df' (e opcionalmente salva CSV).
    """

    # 1) carrega config e arrays base (5 min)
    cfg, times_5m, pv_5m, ld_5m = EnergyEnvConfig.from_parameters_json(parameters_dir)

    # 2) ambiente
    env = EnergyEnvSimpleNP(times_5m, pv_5m, ld_5m, cfg)
    steps = cfg.steps(dt_minutes=dt_minutes, horizon_hours=horizon_hours)
    env = TimeLimit(env, max_episode_steps=steps)

    # 3) ator
    actor, device = load_actor(ckpt_path)

    # 4) reset
    obs, info = env.reset(options={
        "start_date": start_date,
        "dt_minutes": dt_minutes,
        "horizon_hours": horizon_hours,
        "align": align
    })

    # buffers
    T = steps
    times_m = []

    pv_kw     = np.zeros(T, np.float32)
    load_kw   = np.zeros(T, np.float32)  # pode ser NaN se não vier no info
    price     = np.zeros(T, np.float32)
    sin_h     = np.zeros(T, np.float32)
    cos_h     = np.zeros(T, np.float32)
    soc       = np.zeros(T, np.float32)

    action_cmd_kw = np.zeros(T, np.float32)
    p_bess_used_kw = np.zeros(T, np.float32)
    p_bess_eff_kw  = np.zeros(T, np.float32)
    p_bess_cmd_kw  = np.zeros(T, np.float32)
    p_grid_kw      = np.zeros(T, np.float32)

    rewards       = np.zeros(T, np.float32)
    e_buy_kwh     = np.zeros(T, np.float32)
    e_sell_kwh    = np.zeros(T, np.float32)
    violation_kwh = np.zeros(T, np.float32)
    penalty       = np.zeros(T, np.float32)

    dt_h = float(dt_minutes) / 60.0

    for t in range(T):
        # ação determinística do ator
        with torch.no_grad():
            obs_t = torch.as_tensor(obs[None, :], dtype=torch.float32, device=device)
            a, _, a_mean = actor.sample(obs_t, deterministic=True)
            action = a.cpu().numpy()[0]
        action_cmd_kw[t] = float(action[0])

        # step do ambiente
        obs_next, r, terminated, truncated, info = env.step(action)

        # tempo
        times_m.append(_iget(info, ["timestamp_m", "timestamp", "time_m", "time"]))

        # leia séries físicas do 'info'; use fallback apenas quando não houver
        pv_kw[t]   = float(_iget(info, ["pv_kw", "pv_p_kw", "pv"], default=np.nan))
        # load pode não estar no info — será reconstruído depois
        load_kw_val = _iget(info, ["load_kw", "load_p_kw", "load"], default=np.nan)
        try:
            load_kw[t] = float(load_kw_val) if load_kw_val is not None else np.nan
        except Exception:
            load_kw[t] = np.nan

        price[t] = float(_iget(info, ["price", "tariff", "c_energy"], default=np.nan))
        sin_h[t] = float(_iget(info, ["sin_h", "sin_hour"], default=np.nan))
        cos_h[t] = float(_iget(info, ["cos_h", "cos_hour"], default=np.nan))

        soc[t] = _safe_get_soc(info, obs)

        # potências efetivas
        p_used = _iget(info, ["p_bess_eff_kw", "p_bess_kw", "p_bess_cmd_kw"], default=0.0)
        p_bess_used_kw[t] = float(p_used)
        p_bess_eff_kw[t]  = float(info.get("p_bess_eff_kw", p_used))
        p_bess_cmd_kw[t]  = float(info.get("p_bess_cmd_kw", action_cmd_kw[t]))
        p_grid_kw[t]      = float(_iget(info, ["p_grid_kw", "grid_p_kw", "grid_kw", "grid_p"], default=0.0))

        rewards[t] = float(r)

        # energias (kWh)
        if "e_buy_kwh" in info and "e_sell_kwh" in info:
            e_buy_kwh[t]  = float(info["e_buy_kwh"])
            e_sell_kwh[t] = float(info["e_sell_kwh"])
        else:
            grid_p = p_grid_kw[t]
            e_buy_kwh[t]  = max(grid_p,  0.0) * dt_h
            e_sell_kwh[t] = max(-grid_p, 0.0) * dt_h

        violation_kwh[t] = float(info.get("violation_kwh", 0.0))
        penalty[t]       = float(info.get("penalty", 0.0))

        obs = obs_next
        if terminated or truncated:
            break

    # index temporal
    n = len(times_m)
    try:
        times_np = np.array(times_m, dtype="datetime64[m]")
    except Exception:
        times_np = np.array(times_m)

    # DataFrame
    df = pd.DataFrame({
        "time":             times_np,
        "pv_kw":            pv_kw[:n],
        "load_kw":          load_kw[:n],          # pode conter NaN
        "price":            price[:n],
        "sin_h":            sin_h[:n],
        "cos_h":            cos_h[:n],
        "soc":              soc[:n],
        "action_cmd_kw":    action_cmd_kw[:n],
        "p_bess_used_kw":   p_bess_used_kw[:n],
        "p_bess_eff_kw":    p_bess_eff_kw[:n],
        "p_bess_cmd_kw":    p_bess_cmd_kw[:n],
        "p_grid_kw":        p_grid_kw[:n],
        "e_buy_kwh":        e_buy_kwh[:n],
        "e_sell_kwh":       e_sell_kwh[:n],
        "violation_kwh":    violation_kwh[:n],
        "penalty":          penalty[:n],
        "dt_minutes":       np.full(n, dt_minutes, dtype=np.int32),
        "start_date":       np.full(n, start_date),
    }).set_index("time")

    # Reconstrói carga verdadeira pelo balanço quando não houver no info
    # p_grid = load_true - pv + p_bess_used  =>  load_true = pv - p_bess_used + p_grid
    load_true = df["pv_kw"] - df["p_bess_used_kw"] + df["p_grid_kw"]
    # se load_kw existir (não NaN), preferimos o valor do info; caso contrário, usamos o reconstruído
    df["load_true_kw"] = df["load_kw"].where(df["load_kw"].notna(), load_true)

    # erro de balanço (deve ser ~0)
    df["balance_err_kw"] = df["p_grid_kw"] - (df["load_true_kw"] - df["pv_kw"] + df["p_bess_used_kw"])

    # salvar CSV (opcional)
    csv_path = None
    if save_csv_path:
        csv_path = Path(save_csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=True)

    results = {
        "times": times_np,
        "pv_kw": df["pv_kw"].to_numpy(),
        "load_true_kw": df["load_true_kw"].to_numpy(),
        "price": df["price"].to_numpy(),
        "soc": df["soc"].to_numpy(),
        "p_bess_kw": df["p_bess_used_kw"].to_numpy(),
        "p_grid_kw": df["p_grid_kw"].to_numpy(),
        "reward_per_step": df["reward"].to_numpy() if "reward" in df else None,
        "total_reward": float(df["penalty"].sum()*0.0 + df.get("reward", pd.Series(0, index=df.index)).sum()),
        "e_buy_kwh": df["e_buy_kwh"].to_numpy(),
        "e_sell_kwh": df["e_sell_kwh"].to_numpy(),
        "dt_minutes": dt_minutes,
        "horizon_hours": horizon_hours,
        "start_date": start_date,
        "df": df,
        "csv_path": str(csv_path) if csv_path else None,
    }
    return results


# -----------------------
# Plot (operação + SoC no mesmo figure)
# -----------------------

def plot_results(res, save_to: str | None = None, show: bool = True, plot_bad_load=False):
    """
    Plota operação (PV/Load_true/Grid/BESS) e SoC no mesmo figure (2 subplots).
    - BESS e Grid em 'steps-post' (potência constante no intervalo).
    - Se SoC vier fora de [0,1] (ex.: [-1,1]), normaliza para [0,1] apenas para o gráfico.
    - Se plot_bad_load=True e existir 'load_kw' diferente de 'load_true_kw', plota a série antiga pontilhada.
    """
    df = res["df"].copy()
    t = pd.to_datetime(df.index)

    # SoC para [0,1] visualmente
    soc_raw = df["soc"].to_numpy()
    if np.nanmin(soc_raw) < 0.0 or np.nanmax(soc_raw) > 1.0:
        soc_plot = np.clip((soc_raw + 1.0) * 0.5, 0.0, 1.0)
    else:
        soc_plot = soc_raw

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 6.8), sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.0]}
    )

    # --- Topo: operação ---
    ax1.plot(t, df["pv_kw"],   label="PV (kW)")
    ax1.plot(t, df["load_true_kw"], label="Load (kW)")
    if plot_bad_load and "load_kw" in df.columns and not df["load_kw"].equals(df["load_true_kw"]):
        ax1.plot(t, df["load_kw"], linestyle="--", alpha=0.5, label="Load (raw)")
    ax1.plot(t, df["p_grid_kw"], label="Grid (kW)", drawstyle="steps-post")
    ax1.plot(t, df["p_bess_used_kw"], label="BESS (kW)", drawstyle="steps-post")

    ax1.axhline(0.0, lw=1)
    ax1.grid(True, which="both", axis="both", alpha=0.3)
    ax1.set_ylabel("kW")
    ax1.legend(ncol=4, loc="upper right")
    ax1.set_title(f"Operation — {res['start_date']} (dt={res['dt_minutes']} min)")

    # --- Base: SoC ---
    ax2.plot(t, soc_plot, label="SoC")
    ax2.set_ylim(0.0, 1.0)
    ax2.grid(True, which="both", axis="both", alpha=0.3)
    ax2.set_ylabel("fraction")
    ax2.set_xlabel("Time")
    ax2.legend(loc="upper right")

    fig.tight_layout()

    if save_to:
        Path(save_to).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_to, dpi=150)

    if show:
        plt.show()
    else:
        plt.close(fig)


# -----------------------
# Execução direta (exemplo)
# -----------------------

if __name__ == "__main__":
    START_DAY = "2007-01-01"   # escolha um dia presente no dataset
    DT = 15                    # resolução (min)
    HZ = 48                    # 48*15min = 12h — ajuste conforme necessário

    ckpt = "saves/model1/sac_final.pt"

    res = evaluate_day(
        START_DAY,
        dt_minutes=DT,
        horizon_hours=HZ,
        parameters_dir="data",
        ckpt_path=ckpt,
        align="exact",
        save_csv_path=f"saves/model1/eval_{START_DAY}.csv"
    )

    print(f"Total reward on {START_DAY}: {res['total_reward']:.4f}")
    if res['csv_path']:
        print(f"CSV salvo em: {res['csv_path']}")

    out_png = f"saves/model1/eval_{START_DAY}.png"
    plot_results(res, save_to=out_png, show=True, plot_bad_load=True)
