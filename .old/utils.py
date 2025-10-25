# opt/utils.py
# -*- coding: utf-8 -*-
"""
Utility helpers for reading normalized PV/Load profiles from CSV
and aligning them to the model's time grid.

- Two-CSV mode: both files are expected to contain 'timestamp' and 'p_norm' in [0, 1]
  (no clamping is performed here). One CSV is PV, the other is Load.
- Single-CSV mode: the file must contain two distinct normalized columns, one for PV
  and one for Load; you choose the column names via pv_col/load_col parameters.

- Functions return a 'forecasts' dict compatible with teacher.build(...):
    forecasts = {"pv_norm": {datetime: float}, "load_norm": {datetime: float}}
- Also exposes build_time_grid(...) so callers can use the exact same logic
  as the teacher to compute the timeline.
"""

from __future__ import annotations

from typing import Dict, Any, List, Tuple
from datetime import datetime, timedelta

import pandas as pd


def build_time_grid(params: Dict[str, Any], start_dt: datetime) -> List[datetime]:
    """Build ordered timestamps based on horizon_hours and timestep_1_min."""
    horizon_h = int(params["horizon_hours"])
    step_min = int(params["timestep_min"])
    steps = int(horizon_h * 60 // step_min)
    times = [start_dt]
    for _ in range(steps - 1):
        times.append(times[-1] + timedelta(minutes=step_min))
    return times


def _ensure_naive_dtindex(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Drop timezone info to keep timestamps naive (matching teacher expectations)."""
    try:
        if idx.tz is not None:
            return idx.tz_convert(None)
    except AttributeError:
        pass
    return idx


def read_norm_series(
    csv_path: str,
    value_col: str,
    ts_col: str = "timestamp",
) -> pd.Series:
    """
    Read a CSV with columns [timestamp, <value_col>] and return a float Series indexed by timestamp.
    - Typical value_col in two-CSV mode: 'p_norm'
    """
    df = pd.read_csv(csv_path)
    if ts_col not in df.columns or value_col not in df.columns:
        raise ValueError(f"CSV must have columns [{ts_col},{value_col}]. Got {df.columns.tolist()}")
    # Parse and sort timestamps
    df[ts_col] = pd.to_datetime(df[ts_col], utc=False, errors="raise")
    df = df.sort_values(ts_col)
    # Set datetime index
    s = df.set_index(ts_col)[value_col].astype(float)
    # Ensure naive datetime index
    s.index = _ensure_naive_dtindex(pd.DatetimeIndex(s.index))
    # Drop duplicate timestamps keeping the last
    s = s[~s.index.duplicated(keep="last")]
    return s


def _align_series_to_grid(
    s: pd.Series,
    times: List[datetime],
    step_min: int,
) -> pd.Series:
    """
    Align a time series to the exact model time grid:
      1) Resample to grid frequency (if needed) with time interpolation.
      2) Reindex to exact timestamps using nearest within half a step.
      3) Fill any residual gaps with forward/backward fill.
    """
    freq = f"{step_min}min"
    s_res = s.resample(freq).interpolate("time")
    idx = pd.DatetimeIndex(times)
    tol = pd.Timedelta(minutes=max(1, step_min // 2 or 1))
    s_aligned = s_res.reindex(idx, method="nearest", tolerance=tol)
    s_aligned = s_aligned.fillna(method="ffill").fillna(method="bfill")
    return s_aligned.astype(float)


def make_forecasts_from_csv(
    pv_csv: str,
    load_csv: str,
    start_dt: datetime,
    params: Dict[str, Any],
    ts_col: str = "timestamp",
    pv_col: str = "p_norm",    # default: both CSVs expose 'p_norm'
    load_col: str = "p_norm",  # default: both CSVs expose 'p_norm'
) -> Tuple[Dict[str, Dict[datetime, float]], List[datetime]]:
    """
    Build forecasts dict aligned to the model time grid from two CSV files:
    - pv_csv:    columns [timestamp, p_norm]
    - load_csv:  columns [timestamp, p_norm]
    Returns: (forecasts, times)
    """
    times = build_time_grid(params, start_dt)
    step_min = int(params["timestep_min"])

    # Read normalized series (each CSV provides 'p_norm')
    pv_s = read_norm_series(pv_csv, value_col=pv_col, ts_col=ts_col)
    ld_s = read_norm_series(load_csv, value_col=load_col, ts_col=ts_col)

    # Align to grid
    pv_al = _align_series_to_grid(pv_s, times, step_min)
    ld_al = _align_series_to_grid(ld_s, times, step_min)

    forecasts = {
        "pv_norm":   {t.to_pydatetime(): float(pv_al.loc[t]) for t in pv_al.index},
        "load_norm": {t.to_pydatetime(): float(ld_al.loc[t]) for t in ld_al.index},
    }
    return forecasts, times


def make_forecasts_from_single_csv(
    csv_path: str,
    start_dt: datetime,
    params: Dict[str, Any],
    ts_col: str = "timestamp",
    pv_col: str = "pv_norm",     # single-CSV requires two distinct columns
    load_col: str = "load_norm", # customize if your file uses other names
) -> Tuple[Dict[str, Dict[datetime, float]], List[datetime]]:
    """
    Build forecasts from a single CSV with two distinct normalized columns for PV and Load.
    Default column names are 'pv_norm' and 'load_norm' (customize via pv_col/load_col).
    Returns: (forecasts, times)
    """
    df = pd.read_csv(csv_path)
    expected = {ts_col, pv_col, load_col}
    if not expected.issubset(df.columns):
        raise ValueError(f"CSV must have columns {sorted(expected)}. Got {df.columns.tolist()}")

    df[ts_col] = pd.to_datetime(df[ts_col], utc=False, errors="raise")
    df = df.sort_values(ts_col).drop_duplicates(subset=[ts_col], keep="last")
    df.set_index(ts_col, inplace=True)
    df.index = _ensure_naive_dtindex(pd.DatetimeIndex(df.index))

    pv_s = df[pv_col].astype(float)
    ld_s = df[load_col].astype(float)

    times = build_time_grid(params, start_dt)
    step_min = int(params["timestep_1_min"])
    pv_al = _align_series_to_grid(pv_s, times, step_min)
    ld_al = _align_series_to_grid(ld_s, times, step_min)

    forecasts = {
        "pv_norm":   {t.to_pydatetime(): float(pv_al.loc[t]) for t in pv_al.index},
        "load_norm": {t.to_pydatetime(): float(ld_al.loc[t]) for t in ld_al.index},
    }
    return forecasts, times
