# energy_env.py
# Gymnasium env minimal: +Pbess = charge, −Pbess = discharge.
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

class Battery:
    def __init__(self, p):
        eff = float(p.get("efficiency", 0.95))
        self.ηc = float(p.get("eta_c", eff))
        self.ηd = float(p.get("eta_d", eff))
        self.Pmax = float(p["Pmax"])
        self.Emax = float(p["Emax"])
        self.soc_min = float(p.get("soc_min", 0.0))
        self.soc_max = float(p.get("soc_max", 1.0))
        self.β = float(p.get("self_discharging", 0.0))  # per hour
        r = p.get("ramp_kw_per_step", 0.0)
        self.ramp = None if (r is None or float(r) <= 0) else float(r)

    @property
    def Emin(self): return self.Emax * self.soc_min

    @property
    def Ecap(self): return self.Emax * self.soc_max


class Load:
    def __init__(self, p):
        self.Pmax = float(p["Pmax"])
        self.df = pd.read_csv(p["load_file"], index_col=0, parse_dates=True)

    def fload(self, t): return float(self.df.loc[t].values[0]) * self.Pmax


class PV:
    def __init__(self, p):
        self.Pmax = float(p["Pmax"])
        self.df = pd.read_csv(p["generation_file"], index_col=0, parse_dates=True)

    def fpv(self, t): return float(self.df.loc[t].values[0]) * self.Pmax


class EDS:
    def __init__(self, p):
        self.Pmax_in  = float(p.get("Pmax_in",  np.inf))
        self.Pmax_out = float(p.get("Pmax_out", np.inf))


class EnergyEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, param, start_time=None, soc_ini=None, horizon_hours=None, timestep=None,
                 window_size: int = 1):
        super().__init__()
        self.bess = Battery(param["BESS"])
        self.pv   = PV(param["PV"])
        self.load = Load(param["Load"])
        self.eds  = EDS(param["EDS"])

        self.t0 = pd.Timestamp(start_time)
        self.E0 = float(soc_ini) * self.bess.Emax
        self.E  = float(np.clip(self.E0, self.bess.Emin, self.bess.Ecap))

        self.Δt   = int(timestep)                 # minutos por passo
        self.Δt_h = self.Δt / 60.0                # horas por passo
        self.horizon_steps = int(round(float(horizon_hours) * 60.0 / float(self.Δt)))

        self.ceds = dict(param.get("costs", {}).get("EDS", {}))
        self.ceds_max = float(max(self.ceds.values())) if self.ceds else 1.0
        self.c_curt = float(param.get("costs", {}).get("c_pv_curt_per_kwh", 0.0))
        self.c_shed = float(param.get("costs", {}).get("load_shedding", 0.0))

        self.Pnorm = float(param["EDS"].get("Pmax_in", self.eds.Pmax_in))

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        self.K = int(max(1, window_size))
        self.F_BASE = 12
        base_low  = np.array([-1,-1,-1,-1,-1,-1,-1,-1, 0, 0, 0, 0], dtype=np.float32)
        base_high = np.array([ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.tile(base_low, self.K),
            high=np.tile(base_high, self.K),
            dtype=np.float32
        )

        self.t = self.t0
        self.k = 0
        self.E = float(np.clip(self.E, self.bess.Emin, self.bess.Ecap))
        self.Pb_prev = 0.0
        self._rows = []
        self._last_pv_used = self.pv.fpv(self.t0)
        self._last_load_served = self.load.fload(self.t0)

        self._hist = np.zeros((self.K, self.F_BASE), dtype=np.float32)

    # ================= Gym API =================
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        o = options or {}
        if "start_time" in o:
            self.t0 = pd.Timestamp(o["start_time"])
        if "horizon_hours" in o:
            self.horizon_steps = int(round(float(o["horizon_hours"]) * 60.0 / float(self.Δt)))

        self.t = self.t0
        self.k = 0
        self.E = float(np.clip(self.E0, self.bess.Emin, self.bess.Ecap))
        self.Pb_prev = 0.0
        self._rows.clear()
        self._last_pv_used = self.pv.fpv(self.t)
        self._last_load_served = self.load.fload(self.t)

        base = self._obs_base()              
        self._hist[:] = base[None, :]        
        price = self._price(self.t)
        info = {"timestamp": self.t, "tariff": price, "tariff_norm": price / self.ceds_max}
        return self._obs(), info

    def step(self, action):
        a_norm = float(np.asarray(action, dtype=np.float32).reshape(-1)[0])
        a_norm = float(np.clip(a_norm, -1.0, 1.0))
        Pb_des = a_norm * self.bess.Pmax
        Pb_net_cmd = self._apply_ramp(Pb_des, self.Pb_prev)

        load_kw = self.load.fload(self.t)
        pv_kw   = self.pv.fpv(self.t)
        price   = self._price(self.t)

        if Pb_net_cmd >= 0.0:
            Pch_des, Pdis_des = Pb_net_cmd, 0.0
        else:
            Pch_des, Pdis_des = 0.0, -Pb_net_cmd

        Pdis = min(Pdis_des, self._discharge_cap())
        Pch  = min(Pch_des,  self._charge_cap())
        Pb_net_eff = Pch - Pdis  

        A = self._close_balance(load_kw, pv_kw, Pdis, Pch)
        B = self._close_balance(0.0,     pv_kw, Pdis, Pch); B["cost_total"] += self.c_shed * (load_kw * self.Δt_h)
        choose_B = (A["residual_kw"] > 1e-9) or (B["cost_total"] + 1e-12 < A["cost_total"])

        if choose_B:
            served = 0.0
            Pgrid_in, Pgrid_out, Pcurt, residual, step_cost = B["Pgrid_in_kw"], B["Pgrid_out_kw"], B["Curtailment_kw"], B["residual_kw"], B["cost_total"]
            XLOAD = 1
        else:
            served = load_kw
            Pgrid_in, Pgrid_out, Pcurt, residual, step_cost = A["Pgrid_in_kw"], A["Pgrid_out_kw"], A["Curtailment_kw"], A["residual_kw"], A["cost_total"]
            XLOAD = 0

        self.E = float(np.clip(
            (1.0 - self.bess.β * self.Δt_h) * self.E +
            self.Δt_h * (self.bess.ηc * Pch - (1.0 / max(self.bess.ηd, 1e-9)) * Pdis),
            self.bess.Emin, self.bess.Ecap
        ))

        pv_used = max(0.0, pv_kw - Pcurt)
        self._rows.append(dict(
            timestamp=self.t, tariff=price, tariff_norm=price / self.ceds_max,
            Load_kw=load_kw, Load_served_kw=served, Shedding_kw=(load_kw - served),
            PV_kw=pv_kw, PV_used_kw=pv_used, Curtailment_kw=Pcurt,
            P_bess_kw=Pb_net_eff,
            P_bess_discharge_kw=Pdis, P_bess_charge_mag_kw=Pch,
            P_grid_in_kw=Pgrid_in, P_grid_out_kw=Pgrid_out,
            Residual_kw=float(residual), XLOAD=int(XLOAD),
            E_kwh=self.E, SoC_pct=100.0 * self.E / self.bess.Emax,
            cost_total=step_cost, action_used=Pb_net_cmd
        ))

        self.Pb_prev = Pb_net_eff
        self._last_pv_used = pv_used
        self._last_load_served = served

        self.t += pd.Timedelta(minutes=self.Δt)
        self.k += 1

        base_next = self._obs_base()
        if self.K > 1:
            self._hist[:-1, :] = self._hist[1:, :]
        self._hist[-1, :] = base_next

        terminated = False
        truncated  = (self.k >= self.horizon_steps)

        info = {"tariff": price, "tariff_norm": price / self.ceds_max, "row": self._rows[-1]}
        reward = -float(step_cost)
        return self._obs(), reward, terminated, truncated, info


    def _close_balance(self, served_load, pv_kw, Pdis, Pch):
        residual = (served_load + Pch) - (pv_kw + Pdis)
        Pgrid_in = Pgrid_out = Pcurt = 0.0
        if residual >= 0.0:
            take = min(residual, self.eds.Pmax_in) if np.isfinite(self.eds.Pmax_in) else residual
            Pgrid_in += take; residual -= take
        else:
            surplus = -residual
            give = min(surplus, self.eds.Pmax_out) if np.isfinite(self.eds.Pmax_out) else surplus
            Pgrid_out += give; surplus -= give
            Pcurt = max(0.0, surplus); residual = 0.0
        cost = self._price(self.t) * (Pgrid_in * self.Δt_h) + self.c_curt * (Pcurt * self.Δt_h)
        return dict(Pgrid_in_kw=Pgrid_in, Pgrid_out_kw=Pgrid_out, Curtailment_kw=Pcurt,
                    residual_kw=float(residual), cost_total=float(cost))

    def _apply_ramp(self, Pb_des, Pb_prev):
        if self.bess.ramp is None: return Pb_des
        return float(np.clip(Pb_des, Pb_prev - self.bess.ramp, Pb_prev + self.bess.ramp))

    def _discharge_cap(self):
        return self.bess.ηd * max(self.E - self.bess.Emin, 0.0) / max(self.Δt_h, 1e-9)

    def _charge_cap(self):
        return max(self.bess.Ecap - self.E, 0.0) / (max(self.bess.ηc, 1e-9) * max(self.Δt_h, 1e-9))

    def _price(self, ts): return float(self.ceds.get(f"{int(ts.hour):02d}:00", 0.0))

    def _obs_base(self) -> np.ndarray:
        t = self.t
        ms, mc = np.sin(2*np.pi*t.month/12.0), np.cos(2*np.pi*t.month/12.0)
        ds, dc = np.sin(2*np.pi*t.day/31.0),   np.cos(2*np.pi*t.day/31.0)
        hs, hc = np.sin(2*np.pi*t.hour/24.0),  np.cos(2*np.pi*t.hour/24.0)
        ws, wc = np.sin(2*np.pi*t.weekday()/7.0), np.cos(2*np.pi*t.weekday()/7.0)
        Eb = self.E / self.bess.Emax
        c  = self._price(t) / self.ceds_max
        return np.array([ms, mc, ds, dc, hs, hc, ws, wc,
                         self._last_pv_used / self.Pnorm,
                         self._last_load_served / self.Pnorm,
                         Eb, c], dtype=np.float32)

    def _obs(self) -> np.ndarray:
        if self.K == 1:
            return self._obs_base()
        return self._hist.reshape(self.K * self.F_BASE).astype(np.float32)

    def to_dataframe(self):
        return pd.DataFrame(self._rows).set_index("timestamp") if self._rows else pd.DataFrame()
