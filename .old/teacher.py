# -*- coding: utf-8 -*-
"""
On-grid MPC (Pyomo) with constant dt_h from params["time"]["timestep"] (minutes).

- Uses a constant time step dt_h (hours) for the whole horizon.
- All constraints and the objective are defined via def (no lambdas).
- Expects forecasts as dicts keyed by datetime: forecasts["pv_norm"][t], forecasts["load_norm"][t].
- Time-of-use prices come from params["costs"]["EDS"] as {"HH:00": price}.

Notes:
- Comments are in English as requested.
"""

from typing import Dict, Any, List
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import json
import math

from pyomo.environ import (
    ConcreteModel, Set, Param, Var, Reals, NonNegativeReals,
    Objective, Constraint, minimize, value, SolverFactory
)


# --------------------- time grid ---------------------
def build_time_grid(params: Dict[str, Any], start_dt: datetime) -> List[datetime]:
    """Build ordered timestamps based on horizon (hours) and constant step (minutes)."""
    horizon_h = int(params["horizon_hours"])
    step_min = int(params["timestep_min"])
    steps = int(horizon_h * 60 // step_min)
    times = [start_dt]
    delta = timedelta(minutes=step_min)
    for _ in range(steps - 1):
        times.append(times[-1] + delta)
    return times


# --------------------- main class ---------------------
class OnGridMPC:
    def __init__(self, params_path: str = "data/parameters.json"):
        self.params_path = Path(params_path)
        self.params = self._load_params()
        self.model = None
        self.results = None
        self._times: List[datetime] = None

    # --------------------- params ---------------------
    def _load_params(self) -> Dict[str, Any]:
        with open(self.params_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        for sec in ["time", "costs", "BESS", "PV", "Load", "EDS"]:
            if sec not in raw:
                raise KeyError(f"Mandatory section missing from JSON: '{sec}'")

        time  = raw["time"]
        costs = raw["costs"]
        BESS  = raw["BESS"]
        PV    = raw["PV"]
        Load  = raw["Load"]
        EDS   = raw["EDS"] or {}

        p: Dict[str, Any] = {}

        # -------- time --------
        p["horizon_hours"] = int(time["horizon_hours"])
        # Prefer new key "timestep" (minutes); keep legacy "timestep_1_min" as fallback
        if "timestep" in time:
            p["timestep_min"] = int(time["timestep"])
        elif "timestep_1_min" in time:
            p["timestep_min"] = int(time["timestep_1_min"])
        else:
            raise KeyError("Missing time.timestep (minutes) in params.")
        # Constant dt in hours for the whole horizon
        p["dt_h"] = float(p["timestep_min"]) / 60.0

        # -------- costs --------
        # Map "HH:00" -> import price (strict: all used hours must be present)
        p["tou_map"] = dict(costs.get("EDS", {}))
        p["c_bess_deg_per_kwh"] = float(costs.get("bess_degradation_per_kwh", 0.0))

        # -------- BESS --------
        p["P_ch_max_kw"]  = float(BESS["Pmax_kw"])
        p["P_dis_max_kw"] = float(BESS["Pmax_kw"])
        p["E_nom_kwh"]    = float(BESS["Emax_kwh"])
        DoD               = float(BESS.get("DoD_frac", 0.9))
        p["soc_min_frac"] = max(0.0, 1.0 - DoD)
        p["soc_max_frac"] = float(BESS.get("soc_max_frac", 1.0))
        p["eta_c"]        = float(BESS["eta_c"])
        p["eta_d"]        = float(BESS["eta_d"])

        # -------- Grid caps --------
        p["P_grid_import_cap_kw"] = float(EDS.get("Pmax_kw", EDS.get("Pmax", 0.0)))
        p["P_grid_export_cap_kw"] = float(abs(EDS.get("Pmin", 0.0)))

        # -------- Scales (informative) --------
        p["P_PV_nom_kw"] = float(PV["Pmax_kw"])
        p["P_L_nom_kw"]  = float(Load["Pmax_kw"])
        p["ramp_bess_kw_per_step"] = float(BESS.get("ramp_kw_per_step", 0.0))

        return p

    # --------------------- build ---------------------
    def build(self,
              start_dt: datetime,
              forecasts: Dict[str, Dict[datetime, float]],
              E_hat_kwh: float):
        """Build the Pyomo model with constant dt_h and def-based rules."""
        p = self.params

        # Time grid
        times = build_time_grid(p, start_dt)
        self._times = list(times)

        # Successor pairs for energy dynamics
        trans_pairs = list(zip(self._times[:-1], self._times[1:]))

        m = ConcreteModel(name="MPC_OnGrid_Simple")
        m.T = Set(initialize=self._times, ordered=True)
        m.TRANS = Set(initialize=trans_pairs, dimen=2, ordered=True)

        # First step in the horizon
        first_t = self._times[0]

        # ----------------- Parameters -----------------
        m.c_deg     = Param(initialize=p["c_bess_deg_per_kwh"])
        m.E_nom     = Param(initialize=p["E_nom_kwh"])
        m.f_soc_min = Param(initialize=p["soc_min_frac"])
        m.f_soc_max = Param(initialize=p["soc_max_frac"])
        m.P_ch_max  = Param(initialize=p["P_ch_max_kw"])
        m.P_dis_max = Param(initialize=p["P_dis_max_kw"])
        m.eta_c     = Param(initialize=p["eta_c"])
        m.eta_d     = Param(initialize=p["eta_d"])
        m.P_imp_cap = Param(initialize=max(0.0, p["P_grid_import_cap_kw"]))
        m.P_exp_cap = Param(initialize=max(0.0, p["P_grid_export_cap_kw"]))
        m.Ppv_nom   = Param(initialize=p["P_PV_nom_kw"])
        m.Pld_nom   = Param(initialize=p["P_L_nom_kw"])
        m.R_bes     = Param(initialize=float(self.params.get("ramp_bess_kw_per_step", 0.0)))  # unchanged

        # Constant dt (hours) for the whole horizon
        m.dt_h = Param(initialize=float(p["dt_h"]))

        # Profiles
        def load_init(_, t): return float(forecasts["load_norm"][t])
        def pv_init(_, t):   return float(forecasts["pv_norm"][t])
        m.Load_kw = Param(m.T, initialize=load_init)
        m.PV_kw   = Param(m.T, initialize=pv_init)

        # Time-of-use (import) price map c_grid[t] — strict coverage check
        tou_map = p["tou_map"]
        needed_hours = {f"{t.hour:02d}:00" for t in self._times}
        missing = sorted(list(needed_hours - set(tou_map.keys())))
        if missing:
            raise KeyError(
                "Missing TOU price(s) for hour key(s): "
                + ", ".join(missing)
                + ". Provide costs['EDS'] with keys like 'HH:00'."
            )
        price_map: Dict[datetime, float] = {t: float(tou_map[f"{t.hour:02d}:00"]) for t in self._times}
        def cgrid_init(_, t): return price_map[t]
        m.c_grid = Param(m.T, initialize=cgrid_init)

        # ----------------- Variables -----------------
        m.P_ch   = Var(m.T, domain=NonNegativeReals)   # charge power (kW)
        m.P_dis  = Var(m.T, domain=NonNegativeReals)   # discharge power (kW)
        m.P_bess = Var(m.T, domain=Reals)              # net battery power (kW), + = discharge
        m.E      = Var(m.T, domain=NonNegativeReals)   # energy in battery (kWh)

        # Original grid split: import and export (no mutual exclusivity enforced)
        m.P_gin  = Var(m.T, domain=NonNegativeReals)   # grid import (kW)
        m.P_gout = Var(m.T, domain=NonNegativeReals)   # grid export (kW)
        m.P_g    = Var(m.T, domain=Reals)              # net grid power (kW), + = import

        # Linearization of |P_bess|
        m.Pbess_abs = Var(m.T, domain=NonNegativeReals)

        # ----------------- Objective -----------------
        def total_cost_rule(mm):
            """Grid import cost + battery degradation cost (|P_bess|) with constant dt."""
            return sum(
                mm.dt_h * (mm.c_grid[t] * mm.P_gin[t] + mm.c_deg * mm.Pbess_abs[t])
                for t in mm.T
            )
        m.Objective = Objective(rule=total_cost_rule, sense=minimize)

        # ----------------- Constraints -----------------
        def balance_rule(mm, t):
            """Power balance: PV + discharge - charge + grid_in - grid_out = Load."""
            return mm.PV_kw[t] * mm.Ppv_nom + mm.P_dis[t] - mm.P_ch[t] + mm.P_gin[t] - mm.P_gout[t] == mm.Load_kw[t] * mm.Pld_nom
        m.Balance = Constraint(m.T, rule=balance_rule)

        def pbess_link_rule(mm, t):
            """Link net battery power to charge/discharge split."""
            return mm.P_bess[t] == mm.P_dis[t] - mm.P_ch[t]
        m.PbessLink = Constraint(m.T, rule=pbess_link_rule)

        # def abs_pos_rule(mm, t):
        #     """|P_bess| ≥ +P_bess."""
        #     return mm.Pbess_abs[t] >= mm.P_bess[t]
        # m.AbsPos = Constraint(m.T, rule=abs_pos_rule)

        # def abs_neg_rule(mm, t):
        #     """|P_bess| ≥ -P_bess."""
        #     return mm.Pbess_abs[t] >= -mm.P_bess[t]
        # m.AbsNeg = Constraint(m.T, rule=abs_neg_rule)

        def dynamics_rule(mm, t0, t1):
            """Energy dynamics: E[t1] = E[t0] + dt_h*(eta_c*P_ch - (1/eta_d)*P_dis)."""
            return mm.E[t1] == mm.E[t0] + mm.dt_h * (mm.eta_c * mm.P_ch[t0] - (1.0 / mm.eta_d) * mm.P_dis[t0])
        m.Dynamics = Constraint(m.TRANS, rule=dynamics_rule)

        def soc_lo_rule(mm, t):
            """SoC lower bound."""
            return mm.E[t] >= mm.f_soc_min * mm.E_nom
        m.SoC_Lo = Constraint(m.T, rule=soc_lo_rule)

        def soc_hi_rule(mm, t):
            """SoC upper bound."""
            return mm.E[t] <= mm.f_soc_max * mm.E_nom
        m.SoC_Hi = Constraint(m.T, rule=soc_hi_rule)

        def charge_limit_rule(mm, t):
            """Converter max charge power."""
            return mm.P_ch[t] <= mm.P_ch_max
        m.ChargeLimit = Constraint(m.T, rule=charge_limit_rule)

        def discharge_limit_rule(mm, t):
            """Converter max discharge power."""
            return mm.P_dis[t] <= mm.P_dis_max
        m.DischargeLimit = Constraint(m.T, rule=discharge_limit_rule)

        # If R_bess > 0, bound the ramp of net BESS power (unchanged)
        def ramp_up_rule(mm, t0, t1):
            if float(mm.R_bes) <= 0.0:
                return Constraint.Skip
            return mm.P_bess[t1] - mm.P_bess[t0] <= mm.R_bes
        m.RampUp = Constraint(m.TRANS, rule=ramp_up_rule)

        def ramp_dn_rule(mm, t0, t1):
            if float(mm.R_bes) <= 0.0:
                return Constraint.Skip
            return mm.P_bess[t0] - mm.P_bess[t1] <= mm.R_bes
        m.RampDn = Constraint(m.TRANS, rule=ramp_dn_rule)

        # Grid caps (original split)
        def grid_import_cap_rule(mm, t):
            return mm.P_gin[t] <= mm.P_imp_cap
        m.GridImportCap = Constraint(m.T, rule=grid_import_cap_rule)

        def grid_export_cap_rule(mm, t):
            return mm.P_gout[t] <= mm.P_exp_cap
        m.GridExportCap = Constraint(m.T, rule=grid_export_cap_rule)

        def grid_link_rule(mm, t):
            """Link net grid power to import/export split."""
            return mm.P_g[t] == mm.P_gin[t] - mm.P_gout[t]
        m.GridLink = Constraint(m.T, rule=grid_link_rule)

        # Initial SoC (clamped to feasible range)
        E_nom = float(p["E_nom_kwh"])
        E_lo  = float(p["soc_min_frac"]) * E_nom
        E_hi  = float(p["soc_max_frac"]) * E_nom
        E0    = float(E_hat_kwh)
        E0_clamped = min(max(E0, E_lo), E_hi)
        m.E_hat = Param(initialize=E0_clamped)
        def initial_cond_rule(mm):
            """Initial energy condition at the first time step (clamped)."""
            return mm.E[first_t] == mm.E_hat
        m.InitialCond = Constraint(rule=initial_cond_rule)

        self.model = m
        return m

    # --------------------- solve ---------------------
    def solve(self, solver_name="gurobi", tee=True, **solver_opts):
        """Solve the model using the chosen solver and options."""
        solver = SolverFactory(solver_name)
        for k, v in solver_opts.items():
            solver.options[k] = v
        self.results = solver.solve(self.model, tee=tee)
        return self.results

    # --------------------- extract (dict rows) ---------------------
    def extract_solution(self) -> Dict[str, Any]:
        """Return compact timeline with normalized signals + cyclic time features."""
        if self.model is None or self._times is None:
            return {}
        m, ev, eps = self.model, value, 1e-9

        # Denominators: PV/Load from self.params; others from model Params
        pv_nom = max(eps, float(self.params.get("P_PV_nom_kw", 1.0)))
        ld_nom = max(eps, float(self.params.get("P_L_nom_kw", 1.0)))
        p_ch   = max(eps, float(ev(m.P_ch_max)))
        p_dis  = max(eps, float(ev(m.P_dis_max)))
        p_bess = max(eps, max(p_ch, p_dis))
        p_imp  = max(eps, float(ev(m.P_imp_cap)))
        p_exp  = max(eps, float(ev(m.P_exp_cap)))
        e_nom  = max(eps, float(ev(m.E_nom)))

        def cyc(t: datetime) -> Dict[str, float]:
            """Cyclic time features (hour/day-of-week/month)."""
            h = 2 * math.pi * ((t.hour + t.minute / 60 + t.second / 3600) / 24)
            d = 2 * math.pi * (t.weekday() / 7)
            mo = 2 * math.pi * ((t.month - 1) / 12)
            return {
                "hour_sin": math.sin(h), "hour_cos": math.cos(h),
                "dow_sin":  math.sin(d), "dow_cos":  math.cos(d),
                "month_sin": math.sin(mo), "month_cos": math.cos(mo),
            }

        timeline = []
        for t in self._times:
            pv   = ev(m.PV_kw[t] * m.Ppv_nom);   load = ev(m.Load_kw[t] * m.Pld_nom)
            pb   = ev(m.P_bess[t]);  ch   = ev(m.P_ch[t]);  dis  = ev(m.P_dis[t])
            gin  = ev(m.P_gin[t]);   gout = ev(m.P_gout[t]); E = ev(m.E[t])

            row = {
                "timestamp": t.isoformat(),
                "P_bess_kw": pb, "P_ch_kw": ch, "P_dis_kw": dis,
                "P_load_kw": load, "P_pv_kw": pv,
                "P_grid_in_kw": gin, "P_grid_out_kw": gout, "E_kwh": E,
                "pv_norm": pv / pv_nom, "load_norm": load / ld_nom,
                "p_bess_norm": pb / p_bess, "p_ch_norm": ch / p_ch, "p_dis_norm": dis / p_dis,
                "p_grid_in_norm": gin / p_imp, "p_grid_out_norm": gout / p_exp, "E_norm": E / e_nom,
            }
            row.update(cyc(t))
            timeline.append(row)

        return {"decision_time": self._times[0].isoformat(), "timeline": timeline}

    # --------------------- extract (fast array) ---------------------
    def extract_solution_fast(self) -> Dict[str, Any]:
        """Fast extractor: returns a dense float array + column names (no per-step dicts)."""
        if self.model is None or self._times is None:
            return {"decision_time": None, "columns": [], "data": np.empty((0, 0))}

        m, ev, eps = self.model, value, 1e-9

        # Denominators (PV/Load from params; others from model Params)
        pv_nom = max(eps, float(self.params["P_PV_nom_kw"]))
        ld_nom = max(eps, float(self.params["P_L_nom_kw"]))
        p_ch   = max(eps, float(ev(m.P_ch_max)))
        p_dis  = max(eps, float(ev(m.P_dis_max)))
        p_bess = max(eps, max(p_ch, p_dis))
        p_imp  = max(eps, float(ev(m.P_imp_cap)))
        p_exp  = max(eps, float(ev(m.P_exp_cap)))
        e_bess = max(eps, float(ev(m.E_nom)))

        T = len(self._times)

        # Pull raw series in one pass each (less Python overhead)
        pv   = np.fromiter((ev(m.PV_kw[t] * m.Ppv_nom)   for t in self._times), dtype=float, count=T)
        load = np.fromiter((ev(m.Load_kw[t] * m.Pld_nom) for t in self._times), dtype=float, count=T)
        pb   = np.fromiter((ev(m.P_bess[t])  for t in self._times), dtype=float, count=T)
        ch   = np.fromiter((ev(m.P_ch[t])    for t in self._times), dtype=float, count=T)
        dis  = np.fromiter((ev(m.P_dis[t])   for t in self._times), dtype=float, count=T)
        gin  = np.fromiter((ev(m.P_gin[t])   for t in self._times), dtype=float, count=T)
        gout = np.fromiter((ev(m.P_gout[t])  for t in self._times), dtype=float, count=T)
        E    = np.fromiter((ev(m.E[t])       for t in self._times), dtype=float, count=T)

        # Normalized features (no clamping)
        pv_n   = pv   / pv_nom
        load_n = load / ld_nom
        pb_n   = pb   / p_bess
        ch_n   = ch   / p_ch
        dis_n  = dis  / p_dis
        gin_n  = gin  / p_imp
        gout_n = gout / p_exp
        e_n    = E    / e_bess

        # Cyclic time features (vectorized)
        hour = np.fromiter((t.hour + t.minute/60 + t.second/3600 for t in self._times), dtype=float, count=T)
        ha = 2*np.pi*(hour/24)
        dow = np.fromiter((t.weekday() for t in self._times), dtype=float, count=T)
        da = 2*np.pi*(dow/7)
        month = np.fromiter((t.month-1 for t in self._times), dtype=float, count=T)
        ma = 2*np.pi*(month/12)

        hour_sin, hour_cos = np.sin(ha), np.cos(ha)
        dow_sin,  dow_cos  = np.sin(da), np.cos(da)
        month_sin, month_cos = np.sin(ma), np.cos(ma)

        # Stack columns once
        data = np.column_stack([
            pv, load, pb, ch, dis, gin, gout, E,
            pv_n, load_n, pb_n, ch_n, dis_n, gin_n, gout_n, e_n,
            hour_sin, hour_cos, dow_sin, dow_cos, month_sin, month_cos
        ])

        columns = [
            "P_pv_kw","P_load_kw","P_bess_kw","P_ch_kw","P_dis_kw","P_grid_in_kw","P_grid_out_kw","E_kwh",
            "pv_norm","load_norm","p_bess_norm","p_ch_norm","p_dis_norm","p_grid_in_norm","p_grid_out_norm","E_norm",
            "hour_sin","hour_cos","dow_sin","dow_cos","month_sin","month_cos"
        ]

        return {"decision_time": self._times[0].isoformat(), "columns": columns, "data": data}
