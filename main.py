# main.py — parallel sampling (sem NN), enxuto, com dias manuais
from gymnasium.vector import AsyncVectorEnv
from gymnasium.wrappers import TimeLimit
import gymnasium as gym
from environment import EnergyEnvSimpleNP
from environment.config import EnergyEnvConfig
import numpy as np

# Wrapper: injeta SEMPRE as options em todo reset (simples e robusto)
class DefaultResetOptions(gym.Wrapper):
    def __init__(self, env, options: dict):
        super().__init__(env)
        self._options = dict(options)
    def reset(self, *, seed=None, options=None):
        return self.env.reset(seed=seed, options=(options or self._options))

def make_env(times_5m, pv_5m, ld_5m, cfg, dt, hz, start_date, align="next"):
    def _thunk():
        env = EnergyEnvSimpleNP(times_5m, pv_5m, ld_5m, cfg)
        env = TimeLimit(env, max_episode_steps=cfg.steps(dt_minutes=dt, horizon_hours=hz))
        env = DefaultResetOptions(env, {
            "start_date": start_date,
            "dt_minutes": dt,
            "horizon_hours": hz,
            "align": align
        })
        return env
    return _thunk

if __name__ == "__main__":  # necessário no Windows (spawn)
    # 1) Carrega config + base 5-min
    cfg, times_5m, pv_5m, ld_5m = EnergyEnvConfig.from_parameters_json("data")

    # 2) Configura granularidade e horizonte
    dt, hz = 15, 24
    steps = cfg.steps(dt_minutes=dt, horizon_hours=hz)

    # 3) Informe os dias de início manualmente (YYYY-MM-DD)
    user_start_days = ["2006-12-17", "2006-12-18", "2006-12-19", "2006-12-20"]

    # Checagem leve: dia existe no dataset?
    avail = set(map(str, np.unique(times_5m.astype("datetime64[D]"))))
    missing = [d for d in user_start_days if d not in avail]
    if missing:
        dmin = str(times_5m.astype("datetime64[D]").min())
        dmax = str(times_5m.astype("datetime64[D]").max())
        raise ValueError(f"Start day(s) not in dataset: {missing}. Available range: {dmin} → {dmax}")

    # 4) Cria vetor de ambientes (um por dia)
    venv = AsyncVectorEnv([make_env(times_5m, pv_5m, ld_5m, cfg, dt, hz, d) for d in user_start_days])

    # 5) Reset vetorizado (cada subenv injeta suas options)
    obs, infos = venv.reset()

    # 6) Coleta até o TimeLimit
    returns = np.zeros(len(user_start_days), dtype=np.float32)
    for _ in range(steps):
        actions = venv.action_space.sample()  # (n_envs, act_dim)
        obs, rewards, terms, truncs, infos = venv.step(actions)
        returns += rewards.astype(np.float32)

        # opcional: para episódios contínuos, recomece só os que terminaram
        # venv.reset_done()

    print(f"[parallel] dt={dt} min, horizon={hz} h, steps={steps}, n_envs={len(user_start_days)}")
    print("returns per env:", returns)
    print("mean return:", float(returns.mean()))
