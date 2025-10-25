# -*- coding: utf-8 -*-
# SAC fine-tuning with IL warm-start for EnergyEnv
# Additions:
#  (A) env_steps counter (schedules by env-steps, not by global transitions)
#  (B) freeze_alpha_env_steps and warmup_env_steps gating
#  (C) BC weight decays by env_steps (bc_decay_env_steps)
#  (D) demo buffer + mixed sampling (p_demo schedule) for stability
#  (E) offline TD pretrain for critics (optional) with CQL-lite
#  (F) fixes from previous patch: BC shape, correct alpha loss, log_alpha init, done flag

import json, math, random
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from gymnasium.vector import AsyncVectorEnv

from environment import EnergyEnv
from models.model2.model import Hyperparameters, MLPmodel, GaussianTanhActor, QCritic

NAME = "model2"
START_TIME = "2008-12-17 00:00:00"
START_SOC  = 0.5  # fraction of Emax

def set_seed(s=42):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)

class Replay:
    def __init__(self, cap=500_000):
        self.s=[]; self.a=[]; self.r=[]; self.s2=[]; self.d=[]; self.cap=cap
    def add(self, s,a,r,s2,d):
        if len(self.s)>=self.cap:
            self.s.pop(0); self.a.pop(0); self.r.pop(0); self.s2.pop(0); self.d.pop(0)
        self.s.append(s); self.a.append(a); self.r.append(r); self.s2.append(s2); self.d.append(d)
    def __len__(self): return len(self.s)
    def sample(self, bs):
        assert len(self.s) > 0, "Trying to sample from empty buffer"
        idx = np.random.randint(0, len(self.s), size=bs)
        S  = torch.tensor(np.array([self.s[i]  for i in idx]), dtype=torch.float32)
        A  = torch.tensor(np.array([self.a[i]  for i in idx]), dtype=torch.float32).unsqueeze(-1)
        R  = torch.tensor(np.array([self.r[i]  for i in idx]), dtype=torch.float32)
        S2 = torch.tensor(np.array([self.s2[i] for i in idx]), dtype=torch.float32)
        D  = torch.tensor(np.array([self.d[i]  for i in idx]), dtype=torch.float32)
        return S,A,R,S2,D

def sample_mixed(demo_buf: Replay, online_buf: Replay, batch: int, p_demo: float):
    """Sample a batch mixing demo and online transitions."""
    p_demo = float(np.clip(p_demo, 0.0, 1.0))
    len_d = len(demo_buf); len_o = len(online_buf)
    if len_d == 0 and len_o == 0:
        raise RuntimeError("Both demo and online buffers are empty.")
    if len_d == 0:
        return online_buf.sample(batch)
    if len_o == 0 or p_demo >= 1.0:
        return demo_buf.sample(batch)

    bd = int(round(batch * p_demo))
    bd = np.clip(bd, 0, batch)
    bo = batch - bd
    Sd,Ad,Rd,S2d,Dd = demo_buf.sample(bd) if bd > 0 else (None,)*5
    So,Ao,Ro,S2o,Do = online_buf.sample(bo) if bo > 0 else (None,)*5

    if bd == 0: return So,Ao,Ro,S2o,Do
    if bo == 0: return Sd,Ad,Rd,S2d,Dd

    S  = torch.cat([Sd, So], 0)
    A  = torch.cat([Ad, Ao], 0)
    R  = torch.cat([Rd, Ro], 0)
    S2 = torch.cat([S2d, S2o], 0)
    D  = torch.cat([Dd, Do], 0)
    return S,A,R,S2,D

def soft_update(src, tgt, tau):
    for p, pt in zip(src.parameters(), tgt.parameters()):
        pt.data.mul_(1 - tau).add_(tau * p.data)

def main():
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    save_dir = Path(f"saves/{NAME.replace(' ','_')}")
    ckpt_il  = save_dir / "best.pt"
    hp_path  = save_dir / "hp.json"

    cfg = json.load(open(hp_path, "r"))
    hp  = Hyperparameters(cfg)
    rl  = hp.reinforcement_learning

    params_base = json.load(open("data/parameters.json", "r"))
    Pmax = float(params_base["BESS"]["Pmax"])
    Emax = float(params_base["BESS"]["Emax"])  # kept for reference

    # ---- IL warm-start (teacher) ----
    ckpt = torch.load(str(ckpt_il), map_location="cpu")
    input_dim = int(ckpt["input_dim"])
    il = MLPmodel(hp, input_dim=input_dim).to(device)
    il.load_state_dict(ckpt["model_state"]); il.eval()

    # ---- SAC networks ----
    actor = GaussianTanhActor(hp, input_dim=input_dim, action_dim=1).to(device)
    actor.init_from_supervised(il, log_std_init=getattr(rl, "log_std_init", -0.5))

    q1 = QCritic(hp, input_dim, 1).to(device)
    q2 = QCritic(hp, input_dim, 1).to(device)
    q1_t = QCritic(hp, input_dim, 1).to(device); q1_t.load_state_dict(q1.state_dict())
    q2_t = QCritic(hp, input_dim, 1).to(device); q2_t.load_state_dict(q2.state_dict())

    opt_a  = torch.optim.Adam(actor.parameters(), lr=getattr(rl, "lr_actor", 3e-4),  weight_decay=getattr(rl, "weight_decay", 0.0))
    opt_q1 = torch.optim.Adam(q1.parameters(),    lr=getattr(rl, "lr_critic",3e-4), weight_decay=getattr(rl, "weight_decay", 0.0))
    opt_q2 = torch.optim.Adam(q2.parameters(),    lr=getattr(rl, "lr_critic",3e-4), weight_decay=getattr(rl, "weight_decay", 0.0))

    # ---- alpha (temperature) ----
    log_alpha = torch.tensor(
        math.log(max(1e-8, getattr(rl, "alpha_init", 0.1))),
        device=device,
        requires_grad=True
    )
    opt_alpha = torch.optim.Adam([log_alpha], lr=getattr(rl, "lr_alpha", 3e-4))
    target_entropy = float(getattr(rl, "target_entropy", -1.0))  # 1D action

    # ---- hparams ----
    gamma = float(getattr(rl, "gamma", 0.995))
    tau   = float(getattr(rl, "tau",   0.01))
    max_grad_norm = float(getattr(rl, "max_grad_norm", 1.0))
    reward_scale  = float(getattr(rl, "reward_scale", 1.0))
    auto_alpha    = bool(getattr(rl, "auto_alpha", True))

    # schedules by env-steps
    warmup_env_steps       = int(getattr(rl, "warmup_env_steps",  20000))
    freeze_alpha_env_steps = int(getattr(rl, "freeze_alpha_env_steps", 100000))
    bc_decay_env_steps     = int(getattr(rl, "bc_decay_env_steps", 1000000))
    alpha_clamp_min = float(getattr(rl, "alpha_clamp_min", 0.01))
    alpha_clamp_max = float(getattr(rl, "alpha_clamp_max", 0.2))

    nenvs     = int(getattr(rl, "nenvs", 8))
    ndays     = int(getattr(rl, "ndays", 8))
    timestep  = int(getattr(rl, "timestep", 60))  # minutes per step
    n_episodes = int(getattr(rl, "n_episodes", 1))

    updates_per_env_step = float(getattr(rl, "updates_per_env_step", 0.5))
    batch_size = int(getattr(rl, "batch_size", 256))
    policy_delay = int(getattr(rl, "policy_delay", 1))

    # BC schedule (weights)
    bc_coef_max = float(getattr(rl, "bc_coef_max", 0.3))
    bc_coef_min = float(getattr(rl, "bc_coef_min", 0.0))

    # Demo mixing schedule
    p_demo_init            = float(getattr(rl, "p_demo_init", 0.3))
    p_demo_final           = float(getattr(rl, "p_demo_final", 0.0))
    p_demo_decay_env_steps = int(getattr(rl, "p_demo_decay_env_steps", 400000))

    # Pretrain settings
    pretrain_transitions = int(getattr(rl, "pretrain_transitions", 100000))
    pretrain_epochs      = int(getattr(rl, "pretrain_epochs", 10))
    cql_lambda           = float(getattr(rl, "cql_lambda", 0.5))  # 0 disables CQL-lite

    demo_buf   = Replay(cap=int(getattr(rl, "buffer_capacity", 500_000)))
    online_buf = Replay(cap=int(getattr(rl, "buffer_capacity", 500_000)))

    BASE0 = datetime.fromisoformat(START_TIME)

    # ---------- vec env ----------
    def make_env(i):
        def _thunk():
            p = json.loads(json.dumps(params_base))
            return EnergyEnv(
                p,
                start_time=START_TIME,      # will be overwritten by reset(options)
                soc_ini=START_SOC,          # defines self.E0 inside env
                horizon_hours=24,           # placeholder
                timestep=timestep,
            )
        return _thunk
    venv = AsyncVectorEnv([make_env(i) for i in range(nenvs)])

    # ---------- seasonal blocks ----------
    days = [(BASE0 + timedelta(days=i)).replace(hour=0, minute=0, second=0, microsecond=0)
            for i in range(ndays)]
    q, r = divmod(len(days), nenvs)
    sizes = [q + (1 if i < r else 0) for i in range(nenvs)]
    blocks, jdx = [], 0
    for s in sizes:
        blocks.append(days[jdx:jdx+s]); jdx += s

    env_options_base = [{
        "start_time": blocks[i][0].strftime("%Y-%m-%d %H:%M:%S"),
        "horizon_hours": 24 * len(blocks[i]),
    } for i in range(nenvs)]

    # ---------- helpers ----------
    def current_alpha():
        return log_alpha.exp() if auto_alpha else torch.tensor(getattr(rl, "alpha_init", 0.1), device=device)

    def td_target(S2, R, D):
        with torch.no_grad():
            o2 = actor(S2); A2, logp2 = o2["action"], o2["log_prob"]
            alpha = current_alpha()
            tgt = R + gamma * (1.0 - D) * (torch.min(q1_t(S2, A2), q2_t(S2, A2)) - alpha * logp2)
        return tgt

    # ---------- PRETRAIN: collect IL demo and train critics ----------
    env_steps = 0       # counts vectorized env steps
    global_steps = 0    # counts transitions (nenvs per vec step)

    if pretrain_transitions > 0:
        venv.reset_async(seed=None, options=env_options_base)
        s, _ = venv.reset_wait()
        pbar = tqdm(total=pretrain_transitions, desc="[Pretrain] collecting IL transitions", smoothing=0.1)
        while len(demo_buf) < pretrain_transitions:
            st = torch.tensor(s, dtype=torch.float32, device=device)
            with torch.no_grad():
                a_norm = il(st)["action"]
            a_kw = (a_norm.detach().cpu().numpy() * Pmax).astype(np.float32)

            s2, r, term, trunc, info = venv.step(a_kw)

            for i in range(nenvs):
                done_i = bool(term[i] or trunc[i])
                ri = float(r[i]) * reward_scale
                demo_buf.add(s[i], float(a_norm[i].item()), ri, s2[i], float(done_i))

            done = np.logical_or(term, trunc)
            if done.any():
                ropts = [None]*nenvs
                for i in np.where(done)[0]:
                    ropts[i] = env_options_base[i]
                venv.reset_async(seed=None, options=ropts)
                o2, _ = venv.reset_wait()
                try:
                    s2[done] = o2[done]
                except Exception:
                    idx = np.where(done)[0]; jj = 0
                    for i in idx:
                        s2[i] = o2[jj]; jj += 1

            s = s2
            env_steps += 1
            global_steps += nenvs
            pbar.n = min(pretrain_transitions, len(demo_buf)); pbar.refresh()
        pbar.close()

        # TD pretrain (critics only)
        steps = pretrain_epochs * max(1, len(demo_buf)//batch_size)
        pbar = tqdm(total=steps, desc="[Pretrain] TD on critics", smoothing=0.1)
        for _ in range(steps):
            S,A,R,S2,D = demo_buf.sample(batch_size)
            S=S.to(device); A=A.to(device); R=R.to(device); S2=S2.to(device); D=D.to(device)

            target = td_target(S2, R, D)
            # Q1
            loss_q1 = F.mse_loss(q1(S,A), target)
            if cql_lambda > 0.0:
                with torch.no_grad():
                    A_pol = actor(S)["action"]
                loss_q1 = loss_q1 + cql_lambda * (q1(S, A_pol).mean() - q1(S, A).mean())
            opt_q1.zero_grad(); loss_q1.backward()
            if max_grad_norm: torch.nn.utils.clip_grad_norm_(q1.parameters(), max_grad_norm)
            opt_q1.step()
            # Q2
            loss_q2 = F.mse_loss(q2(S,A), target)
            if cql_lambda > 0.0:
                with torch.no_grad():
                    A_pol = actor(S)["action"]
                loss_q2 = loss_q2 + cql_lambda * (q2(S, A_pol).mean() - q2(S, A).mean())
            opt_q2.zero_grad(); loss_q2.backward()
            if max_grad_norm: torch.nn.utils.clip_grad_norm_(q2.parameters(), max_grad_norm)
            opt_q2.step()

            soft_update(q1, q1_t, tau); soft_update(q2, q2_t, tau)
            pbar.update(1)
        pbar.close()

    # ---------- ONLINE FINE-TUNING ----------
    for ep in range(n_episodes):
        venv.reset_async(seed=None, options=env_options_base)
        s, _ = venv.reset_wait()

        finished = np.zeros(nenvs, dtype=bool)
        collected = 0

        ep_R_raw = 0.0
        ep_R_scaled = 0.0

        online_len_before = len(online_buf)
        pbar = tqdm(total=nenvs, desc=f"[EP {ep+1}/{n_episodes}] coleta+treino", smoothing=0.1)

        while collected < nenvs:
            st = torch.tensor(s, dtype=torch.float32, device=device)
            with torch.no_grad():
                use_il = (env_steps <= warmup_env_steps)
                a_norm = il(st)["action"] if use_il else actor(st)["action"]
            a_kw = (a_norm.detach().cpu().numpy() * Pmax).astype(np.float32)

            s2, r, term, trunc, info = venv.step(a_kw)

            for i in range(nenvs):
                done_i = bool(term[i] or trunc[i])
                ri = float(r[i]) * reward_scale
                online_buf.add(s[i], float(a_norm[i].item()), ri, s2[i], float(done_i))
                if not finished[i]:
                    ep_R_raw    += float(r[i])
                    ep_R_scaled += ri

            done = np.logical_or(term, trunc)
            if done.any():
                for i in np.where(done)[0]:
                    if not finished[i]:
                        finished[i] = True
                        collected += 1
                        pbar.set_postfix({
                            "phase":"coleta",
                            "done":f"{collected}/{nenvs}",
                            "buf_on":len(online_buf), "buf_demo":len(demo_buf),
                            "R_raw":f"{ep_R_raw:.1f}", "R_ep":f"{ep_R_scaled:.1f}"
                        })
                        pbar.update(1)
                ropts = [None]*nenvs
                for i in np.where(done)[0]:
                    ropts[i] = env_options_base[i]
                venv.reset_async(seed=None, options=ropts)
                o2, _ = venv.reset_wait()
                try:
                    s2[done] = o2[done]
                except Exception:
                    idx = np.where(done)[0]; jj = 0
                    for i in idx:
                        s2[i] = o2[jj]; jj += 1

            s = s2
            env_steps += 1
            global_steps += nenvs

        # ---- TRAIN ----
        new_online = len(online_buf) - online_len_before
        total_updates = max(1, int(new_online * updates_per_env_step))
        pbar.total = nenvs + total_updates; pbar.refresh()

        avg = {"Lq1":0.0,"Lq2":0.0,"Lact":0.0,"Lalpha":0.0,"alpha":0.0,"H":0.0}; n=0

        # schedule p_demo by env-steps
        if p_demo_decay_env_steps > 0:
            t = np.clip(env_steps / float(p_demo_decay_env_steps), 0.0, 1.0)
            p_demo = p_demo_init * (1.0 - t) + p_demo_final * t
        else:
            p_demo = p_demo_final

        for upd in range(total_updates):
            # sample mixed batch
            S,A,R,S2,D = sample_mixed(demo_buf, online_buf, batch_size, p_demo)
            S=S.to(device); A=A.to(device); R=R.to(device); S2=S2.to(device); D=D.to(device)

            # critics
            with torch.no_grad():
                Y = td_target(S2, R, D)
            loss_q1 = F.mse_loss(q1(S,A), Y); opt_q1.zero_grad(); loss_q1.backward()
            if max_grad_norm: torch.nn.utils.clip_grad_norm_(q1.parameters(), max_grad_norm)
            opt_q1.step()

            loss_q2 = F.mse_loss(q2(S,A), Y); opt_q2.zero_grad(); loss_q2.backward()
            if max_grad_norm: torch.nn.utils.clip_grad_norm_(q2.parameters(), max_grad_norm)
            opt_q2.step()

            # actor (policy_delay) with BC
            update_actor_now = (upd % max(1, policy_delay) == 0)
            if update_actor_now:
                o = actor(S); A_pi, logp = o["action"], o["log_prob"]
                alpha = current_alpha()

                with torch.no_grad():
                    A_il = il(S)["action"]
                if A_il.dim() == 1: A_il = A_il.unsqueeze(-1)
                if A_pi.dim() == 1: A_pi = A_pi.unsqueeze(-1)

                # BC weight by env_steps
                if bc_decay_env_steps > 0:
                    prog = np.clip(env_steps / float(bc_decay_env_steps), 0.0, 1.0)
                    bc_w = bc_coef_max * (1.0 - prog) + bc_coef_min * prog
                else:
                    bc_w = bc_coef_max

                q_pi = torch.min(q1(S, A_pi), q2(S, A_pi))
                bc_loss = F.mse_loss(A_pi, A_il)
                loss_actor = (alpha * logp - q_pi).mean() + bc_w * bc_loss

                opt_a.zero_grad(); loss_actor.backward()
                if max_grad_norm: torch.nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
                opt_a.step()
            else:
                logp = actor(S)["log_prob"].detach()
                loss_actor = torch.tensor(0.0, device=device)

            # α (update gated by env_steps)
            if update_actor_now:
                if auto_alpha and (env_steps > freeze_alpha_env_steps):
                    # SAC-style alpha loss
                    loss_alpha = (log_alpha.exp() * (-logp.detach() - target_entropy)).mean()
                    opt_alpha.zero_grad(); loss_alpha.backward(); opt_alpha.step()
                else:
                    loss_alpha = torch.tensor(0.0, device=device)
                    with torch.no_grad():
                        init_alpha = float(getattr(rl, "alpha_init", 0.1))
                        log_alpha.data.copy_(torch.tensor(math.log(max(1e-8, init_alpha)), device=device))

                with torch.no_grad():
                    log_alpha.data.clamp_(
                        math.log(max(1e-8, alpha_clamp_min)),
                        math.log(max(alpha_clamp_min, alpha_clamp_max))
                    )
            else:
                loss_alpha = torch.tensor(0.0, device=device)

            # targets
            soft_update(q1, q1_t, tau); soft_update(q2, q2_t, tau)

            # logs
            with torch.no_grad():
                avg["Lq1"]+=loss_q1.item(); avg["Lq2"]+=loss_q2.item()
                avg["Lact"]+=loss_actor.item(); avg["Lalpha"]+=loss_alpha.item()
                avg["alpha"]+=current_alpha().item(); avg["H"]+=(-logp).mean().item(); n+=1

            if n and (pbar.n % 20 == 0 or pbar.n+1 == pbar.total):
                pbar.set_postfix({
                    "phase":"treino",
                    "buf_on":len(online_buf), "buf_demo":len(demo_buf),
                    "R_ep":f"{ep_R_scaled:.1f}", "R_raw":f"{ep_R_raw:.1f}",
                    "p_demo":f"{p_demo:.2f}",
                    "Lq1":f"{avg['Lq1']/n:.3f}","Lq2":f"{avg['Lq2']/n:.3f}",
                    "Lact":f"{avg['Lact']/n:.3f}","Lα":f"{avg['Lalpha']/n:.3f}",
                    "α":f"{avg['alpha']/n:.3f}","H":f"{avg['H']/n:.3f}",
                }); avg={k:0.0 for k in avg}; n=0
            pbar.update(1)
        pbar.close()

        if int(getattr(rl, "save_every_episodes", 0)) and ((ep+1) % rl.save_every_episodes == 0):
            save_dir.mkdir(parents=True, exist_ok=True)
            torch.save({"actor_state": actor.state_dict(), "input_dim": input_dim, "cfg": cfg},
                       save_dir / f"sac_actor_ep{ep+1}.pt")

    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"actor_state": actor.state_dict(), "input_dim": input_dim, "cfg": cfg}, save_dir / "sac_actor.pt")
    print(f"SAC concluído. Modelo salvo em {save_dir / 'sac_actor.pt'}")

if __name__ == "__main__":
    set_seed(42)
    main()
