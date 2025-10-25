from datetime import datetime, timedelta
import pyomo.environ as pyo
import pandas as pd
import numpy as np
import math

# Let's create a optimization model using Pyomo for smart home energy management
class Battery:
    def __init__(self, param): # param is a dict
        self.η          = param["efficiency"]
        self.DoD        = param["DoD"]
        self.Pmax       = param["Pmax"]
        self.Emax       = param["Emax"]
        self.soc_min    = param["soc_min"]
        self.soc_max    = param["soc_max"]
        self.β          = param["self_discharging"]
        self.ramp       = param["ramp_kw_per_step"]

class Load:
    def __init__(self, param):
        self.Pmax = param["Pmax"]
        self.df = pd.read_csv(param["load_file"], index_col=0, parse_dates=True)

    def fload(self, t):
        return self.df.loc[t].values[0]

class PV:
    def __init__(self, param):
        self.Pmax = param["Pmax"]
        self.df = pd.read_csv(param["generation_file"], index_col=0, parse_dates=True)
    
    def fpv(self, t):
        return self.df.loc[t].values[0]

class EDS:
    def __init__(self, param):
        self.Pmax_in    = param["Pmax_in"]
        self.Pmax_out   = param["Pmax_out"]


class Teacher:
    def __init__(self, param, pv, load):
        self.bess       = Battery(param["BESS"])
        self.pv         = PV(param["PV"])
        self.load       = Load(param["Load"])
        self.eds        = EDS(param["EDS"])
        self.Δt         = param["time"]["timestep"]
        self.horizon    = param["time"]["horizon_hours"]
        self.ceds       = param["costs"]["EDS"]
        self.cload_shed = param["costs"]["load_shedding"]
        self.Pnorm      = param["EDS"]["Pmax_in"]  # Normalization factor for power


    def build(self, start_time, start_soc):
        Ωt = [start_time + timedelta(minutes=i*self.Δt) for i in range(int(self.horizon*60/self.Δt))]
        self.model      = pyo.ConcreteModel()
        
        # Sets
        self.model.Ωt   = pyo.Set(initialize=Ωt, ordered=True)
        
        # Variables
        self.model.Peds       = pyo.Var(self.model.Ωt, within=pyo.Reals)
        self.model.Peds_in    = pyo.Var(self.model.Ωt, within=pyo.NonNegativeReals, bounds=(0, self.eds.Pmax_in))
        self.model.Peds_out   = pyo.Var(self.model.Ωt, within=pyo.NonNegativeReals, bounds=(0, self.eds.Pmax_out))
        self.model.Ebess      = pyo.Var(self.model.Ωt, within=pyo.NonNegativeReals, bounds=(0, self.bess.Emax))
        self.model.Pbess      = pyo.Var(self.model.Ωt, within=pyo.Reals, bounds=(-self.bess.Pmax, self.bess.Pmax))
        self.model.Pbess_c    = pyo.Var(self.model.Ωt, within=pyo.NonNegativeReals, bounds=(0, self.bess.Pmax))
        self.model.Pbess_d    = pyo.Var(self.model.Ωt, within=pyo.NonNegativeReals, bounds=(0, self.bess.Pmax))
        self.model.γbess_c    = pyo.Var(self.model.Ωt, within=pyo.Binary)
        self.model.γbess_d    = pyo.Var(self.model.Ωt, within=pyo.Binary)
        self.model.PPV        = pyo.Var(self.model.Ωt, within=pyo.NonNegativeReals, bounds=(0, self.pv.Pmax))
        self.model.PLOAD      = pyo.Var(self.model.Ωt, within=pyo.NonNegativeReals, bounds=(0, self.load.Pmax))
        self.model.XPV        = pyo.Var(self.model.Ωt, within=pyo.NonNegativeReals, bounds=(0, 1))
        self.model.XLOAD      = pyo.Var(self.model.Ωt, within=pyo.Binary)
        self.model.PPV_s      = pyo.Param(self.model.Ωt, initialize=lambda m, t: self.pv.fpv(t) * self.pv.Pmax, mutable=True)
        self.model.PLOAD_s    = pyo.Param(self.model.Ωt, initialize=lambda m, t: self.load.fload(t) * self.load.Pmax, mutable=True)


        # Constraints
        def objective_rule(m):
            return sum(
                (self.Δt/60) * (
                    self.model.Peds_in[t] * self.ceds[t.strftime("%H:00")] + 
                    self.cload_shed * self.model.XLOAD[t] * self.model.PLOAD_s[t]
                ) for t in m.Ωt
            )
        self.model.objective = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

        def power_balance_rule(m, t):
            return self.model.Peds[t] + self.model.PPV[t] == self.model.PLOAD[t] + self.model.Pbess[t]
        self.model.power_balance = pyo.Constraint(self.model.Ωt, rule=power_balance_rule)
        
        def pv_curtailement_rule(m, t):
            return self.model.PPV[t] == self.pv.fpv(t) * self.pv.Pmax * (1 - self.model.XPV[t])
        self.model.pv_curtailement = pyo.Constraint(self.model.Ωt, rule=pv_curtailement_rule)

        def load_shedding_rule(m, t):
            return self.model.PLOAD[t] == self.load.fload(t) * self.load.Pmax * (1 - self.model.XLOAD[t])
        self.model.load_shedding = pyo.Constraint(self.model.Ωt, rule=load_shedding_rule)
        
        def eds_rule(m, t):
            return self.model.Peds[t] == self.model.Peds_in[t] - self.model.Peds_out[t]
        self.model.eds_con = pyo.Constraint(self.model.Ωt, rule=eds_rule)

        def bess_rule(m, t):
            return self.model.Pbess[t] == self.model.Pbess_c[t] - self.model.Pbess_d[t]
        self.model.bess_con = pyo.Constraint(self.model.Ωt, rule=bess_rule)

        def energy_bess_rule(m, t):
            if t == m.Ωt.first():
                return self.model.Ebess[t] == self.bess.Emax * start_soc + (self.bess.η * self.model.Pbess_c[t] - (1/self.bess.η) * self.model.Pbess_d[t] - self.bess.β * self.bess.Emax * start_soc) * (self.Δt/60)
            else:
                return self.model.Ebess[t] == self.model.Ebess[m.Ωt.prev(t)] + (self.bess.η * self.model.Pbess_c[t] - (1/self.bess.η) * self.model.Pbess_d[t] - self.bess.β * self.model.Ebess[m.Ωt.prev(t)]) * (self.Δt/60)
        self.model.energy_bess_con = pyo.Constraint(self.model.Ωt, rule=energy_bess_rule)

        def soc_min_rule(m, t):
            return m.Ebess[t] >= self.bess.soc_min * self.bess.Emax
        self.model.soc_min_con = pyo.Constraint(self.model.Ωt, rule=soc_min_rule)

        def soc_max_rule(m, t):
            return m.Ebess[t] <= self.bess.soc_max * self.bess.Emax
        self.model.soc_max_con = pyo.Constraint(self.model.Ωt, rule=soc_max_rule)
        
        def ramp_up_rule(m, t):
            if t == m.Ωt.first():
                return pyo.Constraint.Skip
            return m.Pbess[t] - m.Pbess[m.Ωt.prev(t)] <= self.bess.ramp
        self.model.bess_ramp_up   = pyo.Constraint(self.model.Ωt, rule=ramp_up_rule)

        def ramp_down_rule(m, t):
            if t == m.Ωt.first():
                return pyo.Constraint.Skip
            return m.Pbess[m.Ωt.prev(t)] - m.Pbess[t] <= self.bess.ramp
        self.model.bess_ramp_down = pyo.Constraint(self.model.Ωt, rule=ramp_down_rule)

        def bess_charge_rule(m, t):
            return self.model.Pbess_c[t] <= self.bess.Pmax * self.model.γbess_c[t]
        self.model.bess_charge_con = pyo.Constraint(self.model.Ωt, rule=bess_charge_rule)

        def bess_discharge_rule(m, t):
            return self.model.Pbess_d[t] <= self.bess.Pmax * self.model.γbess_d[t]
        self.model.bess_discharge_con = pyo.Constraint(self.model.Ωt, rule=bess_discharge_rule)

        def bess_binary_rule(m, t):
            return self.model.γbess_c[t] + self.model.γbess_d[t] <= 1
        self.model.bess_binary_con = pyo.Constraint(self.model.Ωt, rule=bess_binary_rule)

        return
    
    def solve(self, solver_name="gurobi"):
        solver = pyo.SolverFactory(solver_name)
        self.results = solver.solve(self.model, tee=True)
        return self.results
    

    def results2dataframe(self):
        df = pd.DataFrame(index=self.model.Ωt)
        # vamos adicionar colunas que falam sobre o horário
        df["month_sin"]     = [np.sin(2 * np.pi * (t.month) / 12) for t in self.model.Ωt]
        df["month_cos"]     = [np.cos(2 * np.pi * (t.month) / 12) for t in self.model.Ωt]
        df["day_sin"]       = [np.sin(2 * np.pi * (t.day) / 31) for t in self.model.Ωt]
        df["day_cos"]       = [np.cos(2 * np.pi * (t.day) / 31) for t in self.model.Ωt]
        df["hour_sin"]      = [np.sin(2 * np.pi * (t.hour) / 24) for t in self.model.Ωt]
        df["hour_cos"]      = [np.cos(2 * np.pi * (t.hour) / 24) for t in self.model.Ωt]
        df["weekday_sin"]   = [np.sin(2 * np.pi * (t.weekday()) / 7) for t in self.model.Ωt]
        df["weekday_cos"]   = [np.cos(2 * np.pi * (t.weekday()) / 7) for t in self.model.Ωt]
        df["load"]          = [pyo.value(self.model.PLOAD[t]) for t in self.model.Ωt]
        df["pv"]            = [pyo.value(self.model.PPV[t]) for t in self.model.Ωt]
        df["pv_available"]  = [pyo.value(self.model.PPV_s[t]) for t in self.model.Ωt]
        df["load_required"] = [pyo.value(self.model.PLOAD_s[t]) for t in self.model.Ωt]
        df["Peds"]          = [pyo.value(self.model.Peds[t]) for t in self.model.Ωt]
        df["Peds_in"]       = [pyo.value(self.model.Peds_in[t]) for t in self.model.Ωt]
        df["Peds_out"]      = [pyo.value(self.model.Peds_out[t]) for t in self.model.Ωt]
        df["Ebess"]         = [pyo.value(self.model.Ebess[t]) for t in self.model.Ωt]
        df["Pbess"]         = [pyo.value(self.model.Pbess[t]) for t in self.model.Ωt]
        df["Pbess_c"]       = [pyo.value(self.model.Pbess_c[t]) for t in self.model.Ωt]
        df["Pbess_d"]       = [pyo.value(self.model.Pbess_d[t]) for t in self.model.Ωt]
        df["γbess_c"]       = [pyo.value(self.model.γbess_c[t]) for t in self.model.Ωt]
        df["γbess_d"]       = [pyo.value(self.model.γbess_d[t]) for t in self.model.Ωt]
        return df
    
    def results2array(self):
        #lets back a array to be used in machine learning. Use the best performance format
        arr = []
        for t in self.model.Ωt:
            arr.append([
                pyo.value(np.sin(2 * np.pi * (t.month) / 12)),
                pyo.value(np.cos(2 * np.pi * (t.month) / 12)),
                pyo.value(np.sin(2 * np.pi * (t.day) / 31)),
                pyo.value(np.cos(2 * np.pi * (t.day) / 31)),
                pyo.value(np.sin(2 * np.pi * (t.hour) / 24)),
                pyo.value(np.cos(2 * np.pi * (t.hour) / 24)),
                pyo.value(np.sin(2 * np.pi * (t.weekday()) / 7)),
                pyo.value(np.cos(2 * np.pi * (t.weekday()) / 7)),
                pyo.value(self.model.PPV[t]),
                pyo.value(self.model.PLOAD[t]),
                pyo.value(self.model.PPV_s[t]),
                pyo.value(self.model.PLOAD_s[t]),
                pyo.value(self.model.Peds[t]),
                pyo.value(self.model.Peds_in[t]),
                pyo.value(self.model.Peds_out[t]),
                pyo.value(self.model.Ebess[t]),
                pyo.value(self.model.Pbess[t]),
                pyo.value(self.model.Pbess_c[t]),
                pyo.value(self.model.Pbess_d[t]),
                pyo.value(self.model.γbess_c[t]),
                pyo.value(self.model.γbess_d[t])
            ])
        return arr

    @property
    def normalized_states(self):
        denom_tar = max(float(v) for v in self.ceds.values())
        Pnorm = float(self.Pnorm)
        Emax  = float(self.bess.Emax)

        arr = []
        for t in self.model.Ωt:
            arr.append([
                math.sin(2*math.pi*t.month/12.0),   math.cos(2*math.pi*t.month/12.0),
                math.sin(2*math.pi*t.day/31.0),     math.cos(2*math.pi*t.day/31.0),
                math.sin(2*math.pi*t.hour/24.0),    math.cos(2*math.pi*t.hour/24.0),
                math.sin(2*math.pi*t.weekday()/7.0),math.cos(2*math.pi*t.weekday()/7.0),

                float(pyo.value(self.model.PPV[t]))   / Pnorm,
                float(pyo.value(self.model.PLOAD[t])) / Pnorm,
                float(pyo.value(self.model.Ebess[t])) / Emax,
                float(self.ceds[t.strftime("%H:00")]) / denom_tar,
            ])
        return arr

    @property
    def normalized_actions(self):
        Pmax = float(self.bess.Pmax)
        return [ float(pyo.value(self.model.Pbess[t])) / Pmax for t in self.model.Ωt ]
