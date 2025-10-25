import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager

# ---------------- Font setup (Gulliver + safe fallbacks) ----------------
plt.rcParams.update({
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.unicode_minus": False,  # use '-' (U+002D) instead of '−' (U+2212) if missing
})

font_path = 'data/Gulliver.otf'
if os.path.exists(font_path):
    font_manager.fontManager.addfont(font_path)
    prop = font_manager.FontProperties(fname=font_path)
    # List to allow automatic fallback for missing glyphs (e.g., minus, en dash)
    plt.rcParams['font.family'] = [prop.get_name(), 'DejaVu Sans', 'sans-serif']
else:
    print("Fonte 'Gulliver.otf' não encontrada, usando Times New Roman.")
    plt.rcParams['font.family'] = ['Times New Roman', 'DejaVu Sans', 'serif']


# ---------------- Plot function ----------------
def plot_solution_from_df(df, e_nom_kwh, figsize=(12, 6.5), save_path=None, show=True, title=None):
    """Plot powers (top) and SoC (bottom) from a dataframe produced by teacher.results2dataframe()."""
    # Ensure datetime index and order
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # Series
    load = pd.to_numeric(df["load"], errors="coerce")
    pv   = pd.to_numeric(df["pv"], errors="coerce")
    grid = (pd.to_numeric(df["Peds"], errors="coerce")
            if "Peds" in df
            else pd.to_numeric(df["Peds_in"], errors="coerce") - pd.to_numeric(df["Peds_out"], errors="coerce"))
    bess = (pd.to_numeric(df["Pbess"], errors="coerce")
            if "Pbess" in df
            else pd.to_numeric(df["Pbess_c"], errors="coerce") - pd.to_numeric(df["Pbess_d"], errors="coerce"))
    E    = pd.to_numeric(df["Ebess"], errors="coerce")

    # Optional originals (dashed traces)
    load_req = pd.to_numeric(df["load_required"], errors="coerce") if "load_required" in df else None
    pv_avail = pd.to_numeric(df["pv_available"],  errors="coerce") if "pv_available"  in df else None

    # Infer bar width in days (matplotlib datetime x-axis uses days)
    if len(df.index) >= 2:
        step_s = float(np.median(np.diff(df.index.values).astype("timedelta64[s]").astype(float)))
    else:
        step_s = 300.0
    width_days = step_s / 86400.0

    # Figure
    fig, (ax_p, ax_soc) = plt.subplots(2, 1, sharex=True, figsize=figsize)
    if title:
        fig.suptitle(title)

    # -------- Top axis: powers --------
    ln_load, = ax_p.step(df.index, load, where="post", label="Load (served, kW)")
    ln_pv,   = ax_p.step(df.index, pv,   where="post", label="PV (used, kW)")
    ax_p.step(df.index, grid, where="post", label="Grid (+Imp/-Exp) (kW)")

    if load_req is not None:
        ax_p.step(df.index, load_req, where="post",
                  linestyle="--", linewidth=1.2, alpha=0.9,
                  color=ln_load.get_color(), label="Load (required, dashed)")
    if pv_avail is not None:
        ax_p.step(df.index, pv_avail, where="post",
                  linestyle="--", linewidth=1.2, alpha=0.9,
                  color=ln_pv.get_color(), label="PV (available, dashed)")

    # BESS bars: positive = charge; negative = discharge
    ax_p.bar(df.index, bess.clip(lower=0), width=width_days, align="edge", alpha=0.6, label="BESS Charge (kW)")
    ax_p.bar(df.index, bess.clip(upper=0), width=width_days, align="edge", alpha=0.6, label="BESS Discharge (kW)")

    ax_p.set_ylabel("Power (kW)")
    ax_p.grid(True, linestyle="--", alpha=0.4)

    # -------- Bottom axis: SoC --------
    soc = (E / float(e_nom_kwh)).clip(0, 1)
    ax_soc.step(df.index, soc, where="post", label="SoC (0-1)")
    ax_soc.set_ylim(0, 1.1)
    ax_soc.set_ylabel("SoC (0-1)")
    ax_soc.grid(True, linestyle="--", alpha=0.4)
    ax_soc.set_xlabel("Time")

    # -------- Legends by axis (inside each axis, bottom side) --------
    # Top axis legend (deduplicated)
    h_p, l_p = ax_p.get_legend_handles_labels()
    handles_p, labels_p, seen = [], [], set()
    for h, l in zip(h_p, l_p):
        if l not in seen:
            handles_p.append(h); labels_p.append(l); seen.add(l)

    ax_p.margins(y=0.15)  # breathing room for the internal legend
    ncol_p = min(len(labels_p), max(2, math.ceil(len(labels_p) / 2)))  # 1–2 lines depending on items
    ax_p.legend(handles_p, labels_p,
                loc="lower left",
                ncol=ncol_p,
                mode="expand",
                bbox_to_anchor=(0.0, 0.00, 1.0, 0.14),  # inside top axis, bottom band
                bbox_transform=ax_p.transAxes,
                frameon=False,
                borderaxespad=0.0)

    # Bottom axis legend (SoC only)
    ax_soc.margins(y=0.12)
    ax_soc.legend(loc="lower left",
                  ncol=1,
                  mode="expand",
                  bbox_to_anchor=(0.0, 0.00, 1.0, 0.12),  # inside bottom axis, bottom band
                  bbox_transform=ax_soc.transAxes,
                  frameon=False,
                  borderaxespad=0.0)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()

    return fig, (ax_p, ax_soc)
