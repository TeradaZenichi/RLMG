# environment/config.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple, Optional
import json
import numpy as np
import pandas as pd


@dataclass
class EnergyEnvConfig:
    # básicos
    dt_minutes: int
    horizon_hours: int
    tariff_by_hour: Dict[str, float]
    feedin_price: float

    # BESS
    E_bess_kwh: float
    P_ch_max_kw: float
    P_dis_max_kw: float
    soc_min: float
    soc_max: float
    soc_init: float
    eta_ch: float
    eta_dis: float
    penalty_violation_kwh: float

    # escalas
    PV_Pmax_kw: float
    Load_Pmax_kw: float

    # treino (opcionais)
    use_shaping: bool = False
    lambda_potential: float = 0.0
    price_weight_by_hour: Dict[str, float] | None = None
    salvage_value_per_kwh: float = 0.0

    # limites de rede / penalizações (opcionais — para ambiente constrained)
    grid_import_kw_max: float = float("inf")
    grid_export_kw_min: float = float("-inf")
    lambda_grid_limits: float = 0.0
    grid_limit_penalty: str = "quadratic"
    grid_limit_huber_delta: float = 0.25
    penalize_grid_limits_in_eval: bool = False

    lambda_clip: float = 0.0
    clip_penalty: str = "quadratic"
    clip_huber_delta: float = 0.25

    def steps(self,
              dt_minutes: Optional[int] = None,
              horizon_hours: Optional[int] = None) -> int:
        """Número de passos para um episódio dado dt (min) e horizonte (h)."""
        dt = int(dt_minutes if dt_minutes is not None else self.dt_minutes)
        hz = int(horizon_hours if horizon_hours is not None else self.horizon_hours)
        if dt <= 0:
            raise ValueError("dt_minutes deve ser > 0")
        return int((hz * 60) // dt)

    @staticmethod
    def _load_any_json_in_dir(dirpath: Path) -> Dict[str, Any]:
        """
        Procura por 'parameters.json' ou 'config.json'; se não achar,
        carrega o primeiro .json do diretório.
        """
        candidates = [
            dirpath / "parameters.json",
            dirpath / "config.json",
        ]
        for c in candidates:
            if c.exists():
                with c.open("r", encoding="utf-8") as f:
                    return json.load(f)
        # fallback: primeiro .json
        for c in dirpath.glob("*.json"):
            with c.open("r", encoding="utf-8") as f:
                return json.load(f)
        raise FileNotFoundError(f"Nenhum arquivo JSON encontrado em {dirpath}")

    @classmethod
    def from_parameters_json(cls, parameters_dir: str) -> Tuple["EnergyEnvConfig", np.ndarray, np.ndarray, np.ndarray]:
        """
        Lê o JSON aninhado e os CSVs de PV/Load, retorna:
        (cfg, times_5m[m], pv_5m[kW], load_5m[kW])
        """
        d = Path(parameters_dir)
        if not d.exists():
            raise FileNotFoundError(parameters_dir)

        data = cls._load_any_json_in_dir(d)

        # --- helpers p/ acessar aninhados com fallback ---
        def get(path: str, default=None):
            cur = data
            for key in path.split("."):
                if isinstance(cur, dict) and key in cur:
                    cur = cur[key]
                else:
                    return default
            return cur

        # Tempo
        dt_minutes = int(get("time.timestep", 5))
        horizon_hours = int(get("time.horizon_hours", 24))

        # Custos
        tariff_by_hour = dict(get("costs.EDS", {}))
        if not tariff_by_hour:
            raise ValueError("costs.EDS ausente ou vazio (mapa HH:00 -> preço).")
        feedin_price = float(get("costs.feedin_price", 0.0))

        # BESS
        E_bess_kwh = float(get("BESS.E_bess_kwh", get("BESS.Emax_kwh", 0.0)))
        if E_bess_kwh <= 0:
            raise ValueError("BESS.E_bess_kwh/Emax_kwh inválido.")
        P_ch_max_kw = float(get("BESS.P_ch_max_kw", get("BESS.Pmax_kw", 0.0)))
        P_dis_max_kw = float(get("BESS.P_dis_max_kw", get("BESS.Pmax_kw", 0.0)))
        soc_min  = float(get("BESS.soc_min", 0.1))
        soc_max  = float(get("BESS.soc_max", 1.0))
        soc_init = float(get("BESS.soc_init", 0.5))
        eta_ch   = float(get("BESS.eta_ch", 1.0))
        eta_dis  = float(get("BESS.eta_dis", 1.0))
        penalty_violation_kwh = float(get("BESS.penalty_violation_kwh", 0.0))

        # Escalas PV/Load
        PV_Pmax_kw   = float(get("PV.Pmax_kw",  np.nan))
        Load_Pmax_kw = float(get("Load.Pmax_kw", np.nan))
        if not np.isfinite(PV_Pmax_kw) or PV_Pmax_kw <= 0:
            PV_Pmax_kw = 1.0
        if not np.isfinite(Load_Pmax_kw) or Load_Pmax_kw <= 0:
            Load_Pmax_kw = 1.0

        # Treino (opcional)
        use_shaping = bool(get("training.use_shaping", False))
        lambda_potential = float(get("training.lambda_potential", 0.0))
        price_weight_by_hour = get("training.price_weight_by_hour", None)
        salvage_value_per_kwh = float(get("training.salvage_value_per_kwh", 0.0))

        # Limites de rede/penalizações (opcionais)
        grid_import_kw_max = float(get("grid_import_kw_max", float("inf")))
        grid_export_kw_min = float(get("grid_export_kw_min", float("-inf")))
        lambda_grid_limits = float(get("lambda_grid_limits", 0.0))
        grid_limit_penalty = str(get("grid_limit_penalty", "quadratic"))
        grid_limit_huber_delta = float(get("grid_limit_huber_delta", 0.25))
        penalize_grid_limits_in_eval = bool(get("penalize_grid_limits_in_eval", False))

        lambda_clip = float(get("lambda_clip", 0.0))
        clip_penalty = str(get("clip_penalty", "quadratic"))
        clip_huber_delta = float(get("clip_huber_delta", 0.25))

        # CSVs
        pv_file = get("io.pv_file", None)
        ld_file = get("io.load_file", None)
        if not pv_file or not ld_file:
            raise ValueError("io.pv_file e/ou io.load_file ausentes no JSON.")

        pv_path = (d / pv_file)
        ld_path = (d / ld_file)
        if not pv_path.exists():
            raise FileNotFoundError(str(pv_path))
        if not ld_path.exists():
            raise FileNotFoundError(str(ld_path))

        def _read_series_5min(path: Path, name_hint: str) -> pd.Series:
            df = pd.read_csv(path)
            # tenta detectar coluna de tempo
            time_col = None
            for c in df.columns:
                lc = c.lower()
                if lc in ("time", "timestamp", "datetime", "date"):
                    time_col = c
                    break
            if time_col is None:
                # tenta usar primeira coluna como tempo se parecer datetime
                c0 = df.columns[0]
                try:
                    pd.to_datetime(df[c0])
                    time_col = c0
                except Exception:
                    pass
            if time_col is None:
                raise ValueError(f"{path.name}: não encontrei coluna de tempo.")

            df[time_col] = pd.to_datetime(df[time_col])
            df = df.set_index(time_col).sort_index()

            # escolhe coluna numérica
            num_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
            if not num_cols:
                raise ValueError(f"{path.name}: não encontrei coluna numérica.")
            # preferências por nome
            pref = {
                "pv": ["pv_kw", "pv", "p_pv", "value", "y"],
                "load": ["load_true_kw", "load_kw", "load", "p_load", "value", "y"],
            }[name_hint]
            chosen = None
            for cand in pref:
                if cand in df.columns:
                    chosen = cand
                    break
            if chosen is None:
                # última coluna numérica
                chosen = num_cols[-1]
            s = df[chosen].astype(float)

            # garante 5 minutos (5min)
            s = s.resample("5min").mean()

            return s

        pv_s = _read_series_5min(pv_path, "pv")
        ld_s = _read_series_5min(ld_path, "load")

        # aplica perfis (opcionais) como escala
        fPV = 1.0
        fL  = 1.0
        try:
            fPV_list = get("profiles.fPV", None)
            if isinstance(fPV_list, (list, tuple)) and len(fPV_list) > 0:
                fPV = float(fPV_list[0])
            fL_list = get("profiles.fL", None)
            if isinstance(fL_list, (list, tuple)) and len(fL_list) > 0:
                fL = float(fL_list[0])
        except Exception:
            pass

        pv_s = pv_s * fPV
        ld_s = ld_s * fL

        # interseção de índices
        idx = pv_s.index.intersection(ld_s.index)
        pv_s = pv_s.reindex(idx).astype(np.float32)
        ld_s = ld_s.reindex(idx).astype(np.float32)

        # constrói times e arrays (datetime64[m])
        times_m = idx.to_numpy(dtype="datetime64[m]")
        pv_5m = pv_s.to_numpy(dtype=np.float32)
        ld_5m = ld_s.to_numpy(dtype=np.float32)

        # configura
        cfg = cls(
            dt_minutes=dt_minutes,
            horizon_hours=horizon_hours,
            tariff_by_hour=tariff_by_hour,
            feedin_price=feedin_price,
            E_bess_kwh=E_bess_kwh,
            P_ch_max_kw=P_ch_max_kw,
            P_dis_max_kw=P_dis_max_kw,
            soc_min=soc_min, soc_max=soc_max, soc_init=soc_init,
            eta_ch=eta_ch, eta_dis=eta_dis,
            penalty_violation_kwh=penalty_violation_kwh,
            PV_Pmax_kw=PV_Pmax_kw, Load_Pmax_kw=Load_Pmax_kw,
            use_shaping=use_shaping,
            lambda_potential=lambda_potential,
            price_weight_by_hour=price_weight_by_hour,
            salvage_value_per_kwh=salvage_value_per_kwh,
            grid_import_kw_max=grid_import_kw_max,
            grid_export_kw_min=grid_export_kw_min,
            lambda_grid_limits=lambda_grid_limits,
            grid_limit_penalty=grid_limit_penalty,
            grid_limit_huber_delta=grid_limit_huber_delta,
            penalize_grid_limits_in_eval=penalize_grid_limits_in_eval,
            lambda_clip=lambda_clip,
            clip_penalty=clip_penalty,
            clip_huber_delta=clip_huber_delta,
        )
        return cfg, times_m, pv_5m, ld_5m
