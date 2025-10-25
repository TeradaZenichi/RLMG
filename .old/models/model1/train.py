# models/model1/train.py
from __future__ import annotations
import os, json, time, random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv
from gymnasium.wrappers import TimeLimit
from tqdm.auto import tqdm

from environment import EnergyEnvSimpleNP
from environment.config import EnergyEnvConfig
from .model import make_sac_nets


# -----------------------
# Utils: seed & replay
# -----------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ReplayBuffer:
    def __init__(self, obs_dim: int, act_dim: int, size: int):
        self.obs = np.zeros((size, obs_dim), dtype=np.float32)
        self.act = np.zeros((size, act_dim), dtype=np.float32)
        self.rew = np.zeros((size,), dtype=np.float32)
        self.next_obs = np.zeros((size, obs_dim), dtype=np.float32)
        self.done = np.zeros((size,), dtype=np.float32)
        self.ptr = 0
        self.size = 0
        self.max_size = size

    def add(self, obs, act, rew, next_obs, done):
        """
        obs: (N, obs_dim)
        act: (N, act_dim)
        rew: (N,)
        next_obs: (N, obs_dim)
        done: (N,)
        """
        n = obs.shape[0]
        idxs = (np.arange(n) + self.ptr) % self.max_size
        self.obs[idxs] = obs
        self.act[idxs] = act
        self.rew[idxs] = rew
        self.next_obs[idxs] = next_obs
        self.done[idxs] = done
        self.ptr = (self.ptr + n) % self.max_size
        self.size = min(self.size + n, self.max_size)

    def sample(self, batch_size: int, device: torch.device):
        idxs = np.random.randint(0, self.size, size=batch_size)
        obs = torch.as_tensor(self.obs[idxs], device=device)
        act = torch.as_tensor(self.act[idxs], device=device)
        rew = torch.as_tensor(self.rew[idxs], device=device)
        next_obs = torch.as_tensor(self.next_obs[idxs], device=device)
        done = torch.as_tensor(self.done[idxs], device=device)
        return obs, act, rew, next_obs, done


# -----------------------
# Env factory (dias manuais)
# -----------------------

class DefaultResetOptions(gym.Wrapper):
    """Sempre injeta as mesmas options em todo reset (simples & robusto)."""
    def __init__(self, env, options: dict):
        super().__init__(env)
        self._options = dict(options)
    def reset(self, *, seed=None, options=None):
        return self.env.reset(seed=seed, options=(options or self._options))


def make_env(times_5m, pv_5m, ld_5m, cfg_env, dt, hz, start_date, align="next"):
    def _thunk():
        env = EnergyEnvSimpleNP(times_5m, pv_5m, ld_5m, cfg_env)
        env = TimeLimit(env, max_episode_steps=cfg_env.steps(dt_minutes=dt, horizon_hours=hz))
        env = DefaultResetOptions(env, {
            "start_date": start_date,
            "dt_minutes": dt,
            "horizon_hours": hz,
            "align": align
        })
        return env
    return _thunk


# -----------------------
# SAC helpers
# -----------------------

@dataclass
class TrainConfig:
    # env
    data_dir: str = "data"
    start_days: Optional[List[str]] = None   # se None, usa os primeiros n_envs dias
    n_envs: int = 4
    dt_minutes: int = 15
    horizon_hours: int = 48
    seed: int = 0

    # paths
    model_dir: str = "models/model1"
    model_json: str = str(Path(__file__).resolve().parent / "model.json")  # robusto relativo a este arquivo
    ckpt_dir: str = "saves/model1"   # destino final

    # training
    total_env_steps: int = 200_000
    eval_every: int = 25_000
    checkpoint_every: int = 50_000
    ep_avg_window: int = 50  # média móvel de recompensas por episódio


def _load_hparams(path_json: str):
    if not os.path.exists(path_json):
        raise FileNotFoundError(f"model.json not found at {path_json}")
    raw = Path(path_json).read_text(encoding="utf-8")
    if not raw.strip():
        raise ValueError(f"model.json is empty at {path_json}")
    try:
        hp = json.loads(raw)
    except json.JSONDecodeError as e:
        snippet = raw[:120].replace("\n", "\\n")
        raise ValueError(f"Invalid JSON at {path_json}: {e}. First chars: {snippet!r}") from e
    return hp.get("network", {}), hp.get("training", {})


@torch.no_grad()
def soft_update(target: nn.Module, online: nn.Module, tau: float):
    for tp, p in zip(target.parameters(), online.parameters()):
        tp.data.mul_(1 - tau).add_(tau * p.data)


def hard_update(target: nn.Module, online: nn.Module):
    target.load_state_dict(online.state_dict())


# -----------------------
# Train loop (SAC)
# -----------------------

def train(cfg: TrainConfig = TrainConfig()):
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- dados e envs base (5 min)
    env_cfg, times_5m, pv_5m, ld_5m = EnergyEnvConfig.from_parameters_json(cfg.data_dir)

    # dias disponíveis (string YYYY-MM-DD)
    days_avail = list(map(str, np.unique(times_5m.astype("datetime64[D]"))))
    if cfg.start_days is None:
        start_days = days_avail[:cfg.n_envs]
    else:
        start_days = cfg.start_days
        missing = [d for d in start_days if d not in days_avail]
        if missing:
            raise ValueError(f"Start day(s) not in dataset: {missing}. Range: {days_avail[0]} → {days_avail[-1]}")
        if len(start_days) < cfg.n_envs:
            raise ValueError(f"Provide at least n_envs({cfg.n_envs}) start days.")

    # passos por episódio (TimeLimit)
    steps_per_ep = env_cfg.steps(cfg.dt_minutes, cfg.horizon_hours)

    # --- vector env
    venv = AsyncVectorEnv([
        make_env(times_5m, pv_5m, ld_5m, env_cfg, cfg.dt_minutes, cfg.horizon_hours, d)
        for d in start_days[:cfg.n_envs]
    ])
    obs, _ = venv.reset()
    n_envs = obs.shape[0]
    obs_dim = obs.shape[1]
    act_space = venv.single_action_space if hasattr(venv, "single_action_space") else venv.action_space
    act_dim = int(np.prod(act_space.shape))
    act_low = act_space.low.astype(np.float32)
    act_high = act_space.high.astype(np.float32)

    # --- redes e hiperparâmetros
    net_hp, tr_hp = _load_hparams(cfg.model_json)
    hidden = net_hp.get("hidden_dims", [256, 256])
    activation = net_hp.get("activation", "silu")
    init = net_hp.get("init", "orthogonal")
    log_std_min = float(net_hp.get("log_std_min", -5.0))
    log_std_max = float(net_hp.get("log_std_max", 2.0))

    nets = make_sac_nets(
        obs_dim, act_dim,
        hidden=hidden,
        activation=activation,
        init=init,
        log_std_bounds=(log_std_min, log_std_max),
        action_low=act_low, action_high=act_high
    )
    actor, critic = nets.actor.to(device), nets.critic.to(device)
    target_critic = type(critic)(obs_dim, act_dim, hidden, activation, init).to(device)
    hard_update(target_critic, critic)

    actor.train()
    critic.train()
    target_critic.eval()  # usado só para alvo; sem dropout/bn, mas é semântico

    # treinos
    gamma = float(tr_hp.get("gamma", 0.99))
    tau = float(tr_hp.get("tau", 0.005))
    lr_actor = float(tr_hp.get("lr_actor", 3e-4))
    lr_critic = float(tr_hp.get("lr_critic", 3e-4))
    lr_alpha = float(tr_hp.get("lr_alpha", 3e-4))
    batch_size = int(tr_hp.get("batch_size", 256))
    buffer_size = int(tr_hp.get("buffer_size", 1_000_000))
    learning_starts = int(tr_hp.get("learning_starts", 5_000))
    gradient_steps = int(tr_hp.get("gradient_steps", 1))
    updates_per_step = int(tr_hp.get("updates_per_step", 1))
    target_update_interval = int(tr_hp.get("target_update_interval", 1))
    max_grad_norm = float(tr_hp.get("max_grad_norm", 10.0))

    alpha_cfg = tr_hp.get("alpha", None)
    target_entropy = tr_hp.get("target_entropy", None)
    auto_alpha = (alpha_cfg is None)
    if target_entropy is None:
        target_entropy = -float(act_dim)

    actor_opt = optim.Adam(actor.parameters(), lr=lr_actor)
    critic_opt = optim.Adam(critic.parameters(), lr=lr_critic)

    if auto_alpha:
        log_alpha = torch.tensor(0.0, device=device, requires_grad=True)  # α = exp(logα)
        alpha_opt = optim.Adam([log_alpha], lr=lr_alpha)
        alpha = log_alpha.exp().item()
    else:
        log_alpha = None
        alpha = float(alpha_cfg)

    # --- buffer & dirs
    rb = ReplayBuffer(obs_dim, act_dim, buffer_size)
    os.makedirs(cfg.ckpt_dir, exist_ok=True)

    # --- treino (episódios síncronos)
    env_steps = 0
    ep_t = 0
    ep_returns = np.zeros(n_envs, dtype=np.float32)
    recent_ep_mean = deque(maxlen=cfg.ep_avg_window)

    pbar = tqdm(total=cfg.total_env_steps, dynamic_ncols=True, desc="Training (env steps)")
    t0 = time.time()

    while env_steps < cfg.total_env_steps:
        # 1) policy
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, device=device)
            if env_steps < learning_starts:
                actions = np.random.uniform(act_low, act_high, size=(n_envs, act_dim)).astype(np.float32)
            else:
                a, _, _ = actor.sample(obs_t, deterministic=False)
                actions = a.cpu().numpy()

        # 2) step
        next_obs, rewards, terms, truncs, infos = venv.step(actions)
        dones = np.logical_and(terms, np.logical_not(truncs)).astype(np.float32)  # done “real”

        # 3) buffer
        rb.add(obs, actions, rewards.astype(np.float32), next_obs, dones)

        # stats
        ep_returns += rewards.astype(np.float32)

        # avanço
        obs = next_obs
        env_steps += n_envs
        ep_t += 1
        pbar.update(n_envs)

        # 4) updates
        if rb.size >= learning_starts:
            updates = gradient_steps * updates_per_step
            for g in range(updates):
                b_obs, b_act, b_rew, b_next, b_done = rb.sample(batch_size, device)

                with torch.no_grad():
                    a_next, logp_next, _ = actor.sample(b_next, deterministic=False)  # [B, A], [B]
                    q1_t, q2_t = target_critic(b_next, a_next)                        # [B], [B]
                    q_min = torch.min(q1_t, q2_t)                                     # [B]
                    v_next = q_min - alpha * logp_next                                # [B]
                    y = b_rew + gamma * (1.0 - b_done) * v_next                       # [B]

                # critics
                q1, q2 = critic(b_obs, b_act)                                         # [B], [B]
                critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)
                critic_opt.zero_grad(set_to_none=True)
                critic_loss.backward()
                nn.utils.clip_grad_norm_(critic.parameters(), max_grad_norm)
                critic_opt.step()

                # actor
                a_pi, logp_pi, _ = actor.sample(b_obs, deterministic=False)
                q1_pi, q2_pi = critic(b_obs, a_pi)
                q_pi = torch.min(q1_pi, q2_pi)
                actor_loss = (alpha * logp_pi - q_pi).mean()
                actor_opt.zero_grad(set_to_none=True)
                actor_loss.backward()
                nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
                actor_opt.step()

                # alpha (auto)
                if auto_alpha:
                    alpha_loss = -(log_alpha * (logp_pi + target_entropy).detach()).mean()
                    alpha_opt.zero_grad(set_to_none=True)
                    alpha_loss.backward()
                    alpha_opt.step()
                    alpha = log_alpha.exp().item()

                if (g % target_update_interval) == 0:
                    soft_update(target_critic, critic, tau)

        # 5) fim de episódio síncrono → reset geral + logging de recompensa
        if ep_t >= steps_per_ep:
            ep_mean = float(ep_returns.mean())
            recent_ep_mean.append(ep_mean)
            avg_show = np.mean(recent_ep_mean) if len(recent_ep_mean) > 0 else ep_mean
            pbar.set_postfix({
                "R̄_ep": f"{ep_mean:.3f}",
                "R̄({})".format(len(recent_ep_mean)): f"{avg_show:.3f}",
                "α": f"{alpha:.3g}",
                "buf": rb.size
            })

            obs, _ = venv.reset()
            ep_returns[:] = 0.0
            ep_t = 0

        # 6) checkpoints
        if cfg.checkpoint_every and (env_steps % cfg.checkpoint_every) == 0:
            ckpt_path = os.path.join(cfg.ckpt_dir, f"sac_step{env_steps}.pt")
            # Salvar listas para compatibilidade com torch.load(weights_only=True)
            torch.save({
                "actor": actor.state_dict(),
                "critic": critic.state_dict(),
                "target_critic": target_critic.state_dict(),
                "alpha": alpha,
                "log_alpha": (log_alpha.detach().cpu() if auto_alpha else None),
                "obs_dim": obs_dim, "act_dim": act_dim,
                "act_low": act_low.tolist(), "act_high": act_high.tolist(),
                "net_hp": net_hp, "tr_hp": tr_hp
            }, ckpt_path)
            pbar.write(f"[ckpt] saved to {ckpt_path}")

    pbar.close()

    # final checkpoint
    final_path = os.path.join(cfg.ckpt_dir, "sac_final.pt")
    torch.save({
        "actor": actor.state_dict(),
        "critic": critic.state_dict(),
        "target_critic": target_critic.state_dict(),
        "alpha": alpha,
        "obs_dim": obs_dim, "act_dim": act_dim,
        "act_low": act_low.tolist(), "act_high": act_high.tolist(),
        "net_hp": net_hp, "tr_hp": tr_hp
    }, final_path)
    print(f"[done] saved final checkpoint to {final_path}")


if __name__ == "__main__":
    train()
