# environment/__init__.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Tuple, Dict, Any
from .config import EnergyEnvConfig

__all__ = ["EnergyEnvSimpleNP", "EnergyEnvConfig"]


class EnergyEnvSimpleNP(gym.Env):
    """
    NumPy-only simple BESS environment (energy-conserving + shaping & salvage + safety layer).

    Base data:
      - Resolução base: 5 minutos.
      - O passo de simulação pode ser {5,10,15,30,60} min (agregando a base).
      - Início por data (YYYY-MM-DD).

    Observação (14 dims, ordem exata):
      [ pv_frac, load_frac, pv_to_load_ratio, net_load_frac, price_norm,
        sin_tod, cos_tod, sin_dow, cos_dow, sin_mon, cos_mon, sin_dom, cos_dom,
        soc ]
        - pv_frac          = PV(kW)/PV_Pmax_kw
        - load_frac        = Load(kW)/Load_Pmax_kw
        - pv_to_load_ratio = clip( PV/Load , 0..3 )
        - net_load_frac    = (Load-PV)/Load_Pmax_kw
        - price_norm       = (price - price_min)/(price_max - price_min)
        - sin/cos 'tod'    = tempo-do-dia por minuto (0..1439)
        - sin/cos 'dow'    = dia da semana (0..6)
        - sin/cos 'mon'    = mês do ano (0..11)
        - sin/cos 'dom'    = dia do mês (0..n_mês-1)
        - soc              = estado de carga (fração)
    """
    metadata = {"render_modes": []}

    # ---------- helpers (estáticos) ----------
    @staticmethod
    def _hours_from_times(t_m: np.ndarray) -> np.ndarray:
        return (t_m.astype("datetime64[h]").astype(int) % 24).astype(np.int32)

    @staticmethod
    def _date_of(t_m: np.ndarray) -> np.ndarray:
        return t_m.astype("datetime64[D]")

    @staticmethod
    def _weekday_of(t_m_scalar: np.datetime64) -> int:
        days = t_m_scalar.astype("datetime64[D]").astype(int)
        return int((days + 3) % 7)  # 1970-01-01 é quinta (=3)

    @staticmethod
    def _minutes_of_day(t_m_scalar: np.datetime64) -> int:
        t_d = t_m_scalar.astype("datetime64[D]")
        return int(((t_m_scalar - t_d).astype("timedelta64[m]").astype(int)) % (24*60))

    @staticmethod
    def _month_index(t_m_scalar: np.datetime64) -> int:
        m = t_m_scalar.astype("datetime64[M]").astype(int)
        return int(m % 12)

    @staticmethod
    def _days_in_month(t_m_scalar: np.datetime64) -> int:
        m0 = t_m_scalar.astype("datetime64[M]")
        m1 = (m0 + np.timedelta64(1, "M")).astype("datetime64[D]")
        m0d = m0.astype("datetime64[D]")
        return int((m1 - m0d).astype("timedelta64[D]").astype(int))

    @staticmethod
    def _day_of_month_index(t_m_scalar: np.datetime64) -> int:
        d = t_m_scalar.astype("datetime64[D]")
        m0d = t_m_scalar.astype("datetime64[M]").astype("datetime64[D]")
        return int((d - m0d).astype("timedelta64[D]").astype(int))

    @staticmethod
    def _aggregate_mean(arr: np.ndarray, k: int) -> np.ndarray:
        n = (arr.size // k) * k
        a = arr[:n].reshape(-1, k)
        return a.mean(axis=1, dtype=np.float32)

    def _grid_limit_penalty_fn(self, excess: float, kind: str, huber_delta: float) -> float:
        """Penalização suave para excesso (kW > 0)."""
        if excess <= 0.0:
            return 0.0
        if kind == "linear":
            return excess
        elif kind == "huber":
            d = max(huber_delta, 1e-9)
            return 0.5 * min(excess, d)**2 + d * max(excess - d, 0.0)
        # default: quadratic
        return excess * excess

    # ---------- construção ----------
    def __init__(self,
                 times_5m_m: np.ndarray,   # datetime64[m]
                 pv_kw_5m: np.ndarray,     # float32
                 load_kw_5m: np.ndarray,   # float32
                 cfg: EnergyEnvConfig):
        super().__init__()
        assert times_5m_m.dtype == "datetime64[m]"
        assert pv_kw_5m.shape == load_kw_5m.shape == times_5m_m.shape
        self.cfg = cfg

        # Flags de treino/avaliação
        self.is_training: bool = False  # altere via set_training_mode() ou reset(options={"mode":...})

        # Física
        self.eta_ch  = float(getattr(self.cfg, "eta_ch", 1.0))
        self.eta_dis = float(getattr(self.cfg, "eta_dis", 1.0))
        self.penalty_violation_kwh = float(getattr(self.cfg, "penalty_violation_kwh", 0.0))

        # Shaping potencial (opcional, só treino)
        self.use_shaping = bool(getattr(self.cfg, "use_shaping", False))
        self.lambda_potential = float(getattr(self.cfg, "lambda_potential", 0.0))

        # Valor de sucata por kWh (só treino, último passo)
        self.salvage_value_per_kwh = float(getattr(self.cfg, "salvage_value_per_kwh", 0.0))

        # Limites de potência na rede (kW). Use +inf / -inf para “sem limite”.
        self.grid_import_kw_max = float(getattr(self.cfg, "grid_import_kw_max", np.inf))
        self.grid_export_kw_min = float(getattr(self.cfg, "grid_export_kw_min", -np.inf))

        # Penalidades (opcionais)
        self.lambda_grid_limits = float(getattr(self.cfg, "lambda_grid_limits", 0.0))
        self.grid_limit_penalty = str(getattr(self.cfg, "grid_limit_penalty", "quadratic"))
        self.grid_limit_huber_delta = float(getattr(self.cfg, "grid_limit_huber_delta", 0.25))
        self.penalize_grid_limits_in_eval = bool(getattr(self.cfg, "penalize_grid_limits_in_eval", False))

        self.lambda_clip = float(getattr(self.cfg, "lambda_clip", 0.0))
        self.clip_penalty = str(getattr(self.cfg, "clip_penalty", "quadratic"))
        self.clip_huber_delta = float(getattr(self.cfg, "clip_huber_delta", 0.25))

        # Base (5-min)
        self.t5  = times_5m_m
        self.pv5 = pv_kw_5m.astype(np.float32, copy=False)
        self.ld5 = load_kw_5m.astype(np.float32, copy=False)

        # Tarifa por hora (24 chaves "HH:00")
        hours = self._hours_from_times(self.t5)
        self.price5 = np.array([self.cfg.tariff_by_hour[f"{h:02d}:00"] for h in hours], dtype=np.float32)

        # Pesos por hora p/ shaping (se não fornecido, usa tarifa normalizada 0..1)
        self.hour_weights = self._build_hour_weights(getattr(self.cfg, "price_weight_by_hour", None))

        # Escalas p/ normalização
        self.pv_kw_max   = float(getattr(self.cfg, "PV_Pmax_kw",  np.nan))
        self.load_kw_max = float(getattr(self.cfg, "Load_Pmax_kw", np.nan))
        if not np.isfinite(self.pv_kw_max)   or self.pv_kw_max   <= 0: self.pv_kw_max   = max(1e-6, float(np.nanmax(self.pv5)))
        if not np.isfinite(self.load_kw_max) or self.load_kw_max <= 0: self.load_kw_max = max(1e-6, float(np.nanmax(self.ld5)))

        # min-max fixo de preço (derivado da própria tarifa por hora)
        tar = np.array([self.cfg.tariff_by_hour[f"{h:02d}:00"] for h in range(24)], dtype=np.float32)
        self.price_min = float(np.min(tar))
        self.price_max = float(np.max(tar))
        self._price_den = max(1e-6, self.price_max - self.price_min)

        # Runtime buffers
        self.dt_minutes = int(self.cfg.dt_minutes)
        self.k = 1
        self.t = self.t5
        self.pv = self.pv5
        self.ld = self.ld5
        self.price = self.price5

        self.idx = 0
        self.T = 0
        self.soc = float(np.clip(self.cfg.soc_init, self.cfg.soc_min, self.cfg.soc_max))

        # Ação: limites de hardware (kW)
        self.action_space = spaces.Box(
            low=np.array([-self.cfg.P_dis_max_kw], dtype=np.float32),
            high=np.array([ self.cfg.P_ch_max_kw], dtype=np.float32),
            dtype=np.float32,
        )

        # Observação: 14 dims com bounds conservadores
        high_vec = self._obs_high_vector()  # shape (14,)
        self.observation_space = spaces.Box(
            low=-high_vec,
            high= high_vec,
            dtype=np.float32,
        )

    # ---------- API auxiliar ----------
    def set_training_mode(self, is_training: bool = True):
        """Liga/desliga shaping e salvage (apenas treino)."""
        self.is_training = bool(is_training)

    def _build_hour_weights(self, mapping: Dict[str, Any] | None) -> np.ndarray:
        """
        Retorna vetor 24 com pesos por hora (0..23).
        Se mapping=None: usa a tarifa min-max normalizada (0..1).
        Se mapping for dict {"HH:00": w} ou {int_hour: w}: usa valores fornecidos.
        """
        w = np.zeros(24, dtype=np.float32)
        if mapping is None:
            p = np.array([self.cfg.tariff_by_hour[f"{h:02d}:00"] for h in range(24)], dtype=np.float32)
            pmin, pmax = float(np.min(p)), float(np.max(p))
            if pmax > pmin:
                w = (p - pmin) / (pmax - pmin)
            else:
                w[:] = 0.0
            return w
        for h in range(24):
            key1 = f"{h:02d}:00"
            if key1 in mapping:
                w[h] = float(mapping[key1])
            elif h in mapping:
                w[h] = float(mapping[h])
            else:
                w[h] = 0.0
        return w

    def _obs_high_vector(self) -> np.ndarray:
        """
        Bounds por componente (low=-high):
        [ pv_frac(1), load_frac(1), pv_to_load_ratio(3), net_load_frac(2),
          price_norm(1), sin_tod(1), cos_tod(1), sin_dow(1), cos_dow(1),
          sin_mon(1), cos_mon(1), sin_dom(1), cos_dom(1), soc(1) ]
        """
        return np.array([1.0, 1.0, 3.0, 2.0,
                         1.0, 1.0, 1.0, 1.0, 1.0,
                         1.0, 1.0, 1.0, 1.0,
                         1.0], dtype=np.float32)

    # ---------- agregação ----------
    def _aggregate_from(self, i0: int, k: int, steps_needed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        start = i0
        total_needed_base = steps_needed * k
        end = min(start + total_needed_base, self.t5.size)
        usable = ((end - start) // k) * k
        end = start + usable
        if usable == 0:
            return (np.empty(0, dtype="datetime64[m]"),
                    np.empty(0, dtype=np.float32),
                    np.empty(0, dtype=np.float32),
                    np.empty(0, dtype=np.float32))
        t_block = self.t5[start:end]
        pv_block = self.pv5[start:end]
        ld_block = self.ld5[start:end]
        pr_block = self.price5[start:end]
        t_out  = t_block[::k]                       # timestamp representativo (primeiro do bloco)
        pv_out = self._aggregate_mean(pv_block, k)
        ld_out = self._aggregate_mean(ld_block, k)
        pr_out = self._aggregate_mean(pr_block, k)  # preço médio no bloco
        return t_out, pv_out, ld_out, pr_out

    def _select_start_index_by_date(self, start_date_str: str, align: str = "next") -> int:
        target = np.datetime64(start_date_str, "D")
        dates = self._date_of(self.t5)
        uniq = np.unique(dates)
        if align == "exact":
            idx = np.nonzero(dates == target)[0]
            if idx.size == 0:
                dmin, dmax = uniq.min(), uniq.max()
                raise ValueError(f"No samples for date {start_date_str}. Available range: {str(dmin)} to {str(dmax)}")
            return int(idx[0])
        i = np.searchsorted(uniq, target, side="left")
        if align == "next":
            if i >= uniq.size: i = uniq.size - 1
        elif align == "prev":
            if i == 0 and uniq[0] > target:
                i = 0
            else:
                if i == uniq.size or uniq[i] != target:
                    i = max(0, i - 1)
        else:
            raise ValueError("align must be one of {'exact','next','prev'}")
        chosen_day = uniq[i]
        j = int(np.nonzero(dates == chosen_day)[0][0])
        return j

    # ---------- Gymnasium API ----------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        opt = options or {}

        # Alternativa de toggle por reset
        mode = opt.get("mode", None)
        if mode == "train":
            self.set_training_mode(True)
        elif mode == "eval":
            self.set_training_mode(False)

        # dt
        dt_minutes = int(opt.get("dt_minutes", self.cfg.dt_minutes))
        if dt_minutes not in (5, 10, 15, 30, 60) or dt_minutes % 5 != 0:
            raise ValueError("dt_minutes must be one of {5,10,15,30,60} and a multiple of 5.")
        self.dt_minutes = dt_minutes
        self.k = dt_minutes // 5

        # start date
        start_date = opt.get("start_date", None)
        align = opt.get("align", "next")
        if start_date is None:
            start_idx_base = 0
        else:
            start_idx_base = self._select_start_index_by_date(str(start_date), align=align)

        # horizonte em passos
        horizon_h = int(opt.get("horizon_hours", self.cfg.horizon_hours))
        steps_needed = int((horizon_h * 60) // self.dt_minutes)

        # agrega
        t_out, pv_out, ld_out, pr_out = self._aggregate_from(start_idx_base, self.k, steps_needed)
        if t_out.size == 0:
            dmin, dmax = self._date_of(self.t5).min(), self._date_of(self.t5).max()
            raise ValueError(
                "Not enough data to build the requested horizon from the selected start date. "
                f"Available date range: {str(dmin)} to {str(dmax)}"
            )

        # runtime
        self.t = t_out
        self.pv = pv_out
        self.ld = ld_out
        self.price = pr_out
        self.T = self.t.size

        self.idx = 0
        self.soc = float(np.clip(self.cfg.soc_init, self.cfg.soc_min, self.cfg.soc_max))
        return self._obs_at(self.idx), {}

    def step(self, action):
        # fim seguro
        if self.idx >= self.T:
            info = {"warning": "step_called_after_episode_end"}
            return self._obs_at(self.T - 1), 0.0, False, True, info

        # comando (kW) limitado por hardware
        p_cmd = float(np.clip(action[0], -self.cfg.P_dis_max_kw, self.cfg.P_ch_max_kw))

        # estados atuais
        pv = float(self.pv[self.idx])
        ld = float(self.ld[self.idx])
        price = float(self.price[self.idx])
        t_m = self.t[self.idx]

        # tempo & capacidade
        dt_h = max(self.dt_minutes / 60.0, 1e-9)
        E = max(self.cfg.E_bess_kwh, 1e-9)
        soc = float(self.soc)

        # folgas de energia (lado bateria)
        E_dn = (soc - self.cfg.soc_min) * E
        E_up = (self.cfg.soc_max - soc) * E

        # pedido no lado AC (antes de qualquer limitação por SoC)
        e_req_ac = p_cmd * dt_h

        # --------- Limites físicos por passo (p_eff em kW) independentemente de p_cmd ---------
        # Máximo AC de carga (kW) limitado por SoC/eficiência
        p_eff_max_physical = (E_up / max(self.eta_ch, 1e-9)) / dt_h
        # Mínimo AC de descarga (kW negativo)
        p_eff_min_physical = - (self.eta_dis * E_dn) / dt_h
        # Limites de hardware (ação em kW, lado AC)
        p_eff_max_hw = float(self.cfg.P_ch_max_kw)
        p_eff_min_hw = -float(self.cfg.P_dis_max_kw)

        # --------- Efetivação pelo SoC (sem safety layer ainda): p_eff_phy ---------
        if e_req_ac >= 0.0:  # carga
            feasible_ac_max = E_up / max(self.eta_ch, 1e-9)  # kWh no lado AC
            e_ac_eff = min(e_req_ac, feasible_ac_max)
            violation_kwh = max(0.0, e_req_ac - feasible_ac_max)
        else:               # descarga
            feasible_ac_abs = self.eta_dis * E_dn            # kWh (módulo) no lado AC
            e_ac_eff = -min(abs(e_req_ac), feasible_ac_abs)
            violation_kwh = max(0.0, abs(e_req_ac) - feasible_ac_abs)
        p_eff_phy = e_ac_eff / dt_h  # kW (lado AC) após SoC

        # Clip por hardware (garante coerência com action_space)
        p_eff_phy = float(np.clip(p_eff_phy, p_eff_min_hw, p_eff_max_hw))

        # --------- SAFETY LAYER: projeção para satisfazer limites de rede por passo ---------
        net = ld - pv  # parte "fixa" do balanço
        lo_grid, hi_grid = self.grid_export_kw_min, self.grid_import_kw_max  # grid_p ∈ [lo_grid, hi_grid]
        # Faixa de p_eff imposta pelos limites de rede
        p_eff_min_grid = lo_grid - net
        p_eff_max_grid = hi_grid - net

        # Interseção de TODAS as faixas
        lo_eff = max(p_eff_min_physical, p_eff_min_hw, p_eff_min_grid)
        hi_eff = min(p_eff_max_physical, p_eff_max_hw, p_eff_max_grid)

        safety_infeasible = False
        if lo_eff <= hi_eff:
            p_eff_star = float(np.clip(p_eff_phy, lo_eff, hi_eff))
        else:
            # Interseção vazia: fisicamente impossível satisfazer a rede neste passo.
            # Opta por respeitar a física (SoC/hardware) e contabiliza violação de rede.
            safety_infeasible = True
            p_eff_star = float(np.clip(p_eff_phy, p_eff_min_physical, p_eff_max_physical))
            p_eff_star = float(np.clip(p_eff_star, p_eff_min_hw, p_eff_max_hw))

        # Métrica de clipping (educa a política)
        clip_kw = abs(p_eff_star - p_eff_phy)

        # Balanço de rede e custo econômico (com p_eff já projetado)
        grid_p = ld - pv + p_eff_star  # >0 import; <0 export
        e_buy  = max(grid_p, 0.0) * dt_h
        e_sell = max(-grid_p, 0.0) * dt_h
        reward_energy = - (e_buy * price - e_sell * self.cfg.feedin_price)

        # Penalização por tentativa inviável no SoC (medida no pedido AC original)
        reward_penalty = - self.penalty_violation_kwh * violation_kwh

        # Atualização de SoC (usando p_eff_star)
        e_ac_eff_star = p_eff_star * dt_h
        if e_ac_eff_star >= 0.0:  # carga
            e_batt_eff = self.eta_ch * e_ac_eff_star
            delta_soc = (e_batt_eff / E)
        else:                     # descarga
            e_batt_eff = abs(e_ac_eff_star) / max(self.eta_dis, 1e-9)
            delta_soc = (-e_batt_eff / E)
        self.soc = float(np.clip(soc + delta_soc, self.cfg.soc_min, self.cfg.soc_max))

        # próxima observação
        next_idx = self.idx + 1
        truncated = next_idx >= self.T
        obs_next_idx = self.idx if truncated else next_idx
        obs_next = self._obs_at(obs_next_idx)

        # --- shaping (apenas treino) ---
        reward_shaping = 0.0
        if self.is_training and self.use_shaping and self.lambda_potential != 0.0:
            hour_now  = int(t_m.astype("datetime64[h]").astype(int) % 24)
            phi_now   = self.lambda_potential * float(self.hour_weights[hour_now]) * soc
            t_next = self.t[obs_next_idx]
            hour_next = int(t_next.astype("datetime64[h]").astype(int) % 24)
            phi_next  = self.lambda_potential * float(self.hour_weights[hour_next]) * self.soc
            reward_shaping = (phi_next - phi_now)

        # --- salvage (apenas treino, último passo) ---
        reward_salvage = 0.0
        if self.is_training and truncated and self.salvage_value_per_kwh != 0.0:
            reward_salvage = self.salvage_value_per_kwh * E * self.soc

        # ----- Penalização por ultrapassar limites de rede -----
        ex_imp = ex_exp = 0.0
        if np.isfinite(self.grid_import_kw_max):
            ex_imp = max(0.0, grid_p - self.grid_import_kw_max)
        if np.isfinite(self.grid_export_kw_min):
            ex_exp = max(0.0, self.grid_export_kw_min - grid_p)

        pen_imp = self._grid_limit_penalty_fn(ex_imp, self.grid_limit_penalty, self.grid_limit_huber_delta)
        pen_exp = self._grid_limit_penalty_fn(ex_exp, self.grid_limit_penalty, self.grid_limit_huber_delta)
        reward_grid_limits = - self.lambda_grid_limits * (pen_imp + pen_exp) * dt_h

        # ----- Penalização pelo clipping do safety layer -----
        pen_clip = self._grid_limit_penalty_fn(clip_kw, self.clip_penalty, self.clip_huber_delta)
        reward_clip = - self.lambda_clip * pen_clip * dt_h

        # recompensa retornada
        if self.is_training:
            reward = float(reward_energy + reward_penalty + reward_shaping + reward_salvage
                           + reward_grid_limits + reward_clip)
        else:
            reward = float(reward_energy + reward_penalty
                           + (reward_grid_limits if self.penalize_grid_limits_in_eval else 0.0)
                           + (reward_clip if self.penalize_grid_limits_in_eval else 0.0))

        # avança o índice
        self.idx = next_idx if next_idx <= self.T else self.T
        terminated = False

        # sinais de hora (para o script de avaliação)
        tod_min = self._minutes_of_day(t_m)
        sin_h = float(np.sin(2 * np.pi * (tod_min / 1440.0)))
        cos_h = float(np.cos(2 * np.pi * (tod_min / 1440.0)))

        info = {
            "timestamp_m": t_m,
            "price": price,

            # Publica PV/Load
            "pv_kw": pv,
            "load_kw": ld,

            # BESS / GRID (antes e depois do safety)
            "p_bess_cmd_kw": p_cmd,
            "p_bess_eff_before_clip_kw": p_eff_phy,
            "p_bess_eff_kw": p_eff_star,
            "p_grid_before_clip_kw": ld - pv + p_eff_phy,
            "p_grid_kw": grid_p,

            # Energia e penalizações
            "e_buy_kwh": e_buy,
            "e_sell_kwh": e_sell,
            "violation_kwh": violation_kwh,
            "grid_import_kw_max": self.grid_import_kw_max,
            "grid_export_kw_min": self.grid_export_kw_min,
            "grid_excess_import_kw": ex_imp,
            "grid_excess_export_kw": ex_exp,
            "clip_kw": clip_kw,

            # Recompensas decompostas
            "reward_energy": reward_energy,
            "reward_penalty": reward_penalty,
            "reward_shaping": reward_shaping,
            "reward_salvage": reward_salvage,
            "reward_grid_limits": reward_grid_limits,
            "reward_clip": reward_clip,
            "reward_returned": reward,
            "is_training": self.is_training,
            "safety_infeasible": safety_infeasible,

            # Auxiliares de hora
            "sin_h": sin_h,
            "cos_h": cos_h,

            # Estado
            "soc": self.soc,
        }
        return obs_next, reward, terminated, truncated, info

    # -------- observation helpers --------
    def _obs_at(self, i: int) -> np.ndarray:
        t_m = self.t[i]

        # Tempo
        tod_min = self._minutes_of_day(t_m)            # 0..1439
        dow     = self._weekday_of(t_m)                # 0..6
        mon_idx = self._month_index(t_m)               # 0..11
        dim     = self._day_of_month_index(t_m)        # 0..(n_mês-1)
        n_m     = self._days_in_month(t_m)             # 28..31

        sin_tod = np.sin(2 * np.pi * (tod_min / 1440.0))
        cos_tod = np.cos(2 * np.pi * (tod_min / 1440.0))
        sin_dow = np.sin(2 * np.pi * (dow / 7.0))
        cos_dow = np.cos(2 * np.pi * (dow / 7.0))
        sin_mon = np.sin(2 * np.pi * (mon_idx / 12.0))
        cos_mon = np.cos(2 * np.pi * (mon_idx / 12.0))
        sin_dom = np.sin(2 * np.pi * (dim / max(1.0, float(n_m))))
        cos_dom = np.cos(2 * np.pi * (dim / max(1.0, float(n_m))))

        # Normalizações & relações PV↔demanda
        pv_kw   = float(self.pv[i])
        load_kw = float(self.ld[i])
        price   = float(self.price[i])

        pv_frac        = pv_kw   / max(self.pv_kw_max,   1e-6)
        load_frac      = load_kw / max(self.load_kw_max, 1e-6)
        net_load_frac  = (load_kw - pv_kw) / max(self.load_kw_max, 1e-6)

        ratio_clip = 3.0
        pv_to_load_ratio = pv_kw / max(load_kw, 1e-6)
        pv_to_load_ratio = float(np.clip(pv_to_load_ratio, 0.0, ratio_clip))

        price_norm = (price - self.price_min) / self._price_den

        return np.array([
            pv_frac, load_frac, pv_to_load_ratio, net_load_frac, price_norm,
            sin_tod, cos_tod, sin_dow, cos_dow, sin_mon, cos_mon, sin_dom, cos_dom,
            self.soc
        ], dtype=np.float32)

    def _obs(self) -> np.ndarray:
        i = min(self.idx, max(self.T - 1, 0))
        return self._obs_at(i)
