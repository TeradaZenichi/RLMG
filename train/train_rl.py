# train/train_rl.py
from train.replaybuffer import ReplayBuffer
from gymnasium.vector import AsyncVectorEnv
from datetime import datetime, timedelta
from collections import deque
from typing import Optional
from copy import deepcopy
from pathlib import Path
import torch.nn.functional as F
import pandas as pd
import numpy as np
import torch
import json
from tqdm.auto import tqdm
from environment import EnergyEnv
import random

# ---------------- utils ----------------
OPTIMIZERS = {
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
    "sgd": torch.optim.SGD,
}

def set_seed_everywhere(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass

def lin_anneal(start: float, end: float, cur_step: int, total_steps: int) -> float:
    if total_steps <= 0:
        return start
    frac = max(0.0, min(1.0, cur_step / float(total_steps)))
    return (1.0 - frac) * float(start) + frac * float(end)

def critic_loss(q1, q2, target, rl):
    # Usa Huber com delta (beta) configurável; cai para MSE se use_huber=False
    if bool(getattr(rl, "use_huber", True)):
        try:
            l1 = F.smooth_l1_loss(q1, target, beta=float(getattr(rl, "huber_delta", 1.0)))
            l2 = F.smooth_l1_loss(q2, target, beta=float(getattr(rl, "huber_delta", 1.0)))
        except TypeError:
            l1 = F.smooth_l1_loss(q1, target)
            l2 = F.smooth_l1_loss(q2, target)
        return l1 + l2
    else:
        return F.mse_loss(q1, target) + F.mse_loss(q2, target)

# ---------------- envs ----------------
def make_vec_envs(rl, params, start_time, start_soc):
    nenvs = min(rl.nenvs, max(1, rl.ndays))
    base = pd.Timestamp(start_time).normalize()
    days = [(base + timedelta(days=i)).strftime("%Y-%m-%d %H:%M:%S") for i in range(rl.ndays)]
    q, r = divmod(rl.ndays, nenvs)
    sizes = [q + (1 if i < r else 0) for i in range(nenvs)]
    blocks, j = [], 0
    for ssz in sizes:
        blocks.append(days[j:j + ssz])
        j += ssz
    env_opts = [{"start_time": b[0], "horizon_hours": 24 * len(b)} for b in blocks]

    def make_env(i):
        def _thunk(i=i):
            p = json.loads(json.dumps(params))
            return EnergyEnv(
                p,
                start_time=env_opts[i]["start_time"],
                soc_ini=start_soc,
                horizon_hours=env_opts[i]["horizon_hours"],
                timestep=rl.timestep,
            )
        return _thunk

    envs = [make_env(i) for i in range(nenvs)]
    return AsyncVectorEnv(envs), env_opts

# ---------------- models ----------------
def make_actor_min(venv, hp, actor_arch, model_arch, model_weights):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NF = int(venv.single_observation_space.shape[0])
    NA = int(venv.single_action_space.shape[0])
    actor = actor_arch(hp, input_dim=NF, action_dim=NA).to(device)
    if model_weights:
        try:
            ckpt = torch.load(model_weights, map_location=device)
            il = ckpt.get("model_state") if isinstance(ckpt, dict) else None
            if il:
                sd = actor.state_dict()
                tr = {k: v for k, v in il.items() if k in sd and sd[k].shape == v.shape}
                sd.update(tr)
                if "head.weight" in il and "mu_head.weight" in sd and sd["mu_head.weight"].shape == il["head.weight"].shape:
                    sd["mu_head.weight"] = il["head.weight"]
                if "head.bias" in il and "mu_head.bias" in sd and sd["mu_head.bias"].shape == il["head.bias"].shape:
                    sd["mu_head.bias"] = il["head.bias"]
                actor.load_state_dict(sd, strict=False)
        except Exception:
            pass
    return actor

def make_critics_min(venv, hp, twincritic_cls):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NF = int(venv.single_observation_space.shape[0])
    NA = int(venv.single_action_space.shape[0])
    q = twincritic_cls(hp, state_dim=NF, action_dim=NA).to(device)
    q_targ = twincritic_cls(hp, state_dim=NF, action_dim=NA).to(device)
    q_targ.load_state_dict(q.state_dict())

    def soft_update(src, dst, tau):
        with torch.no_grad():
            for p, pt in zip(src.parameters(), dst.parameters()):
                pt.mul_(1.0 - tau)
                pt.add_(tau * p)
    return q, q_targ, soft_update

# ---------------- rollout ----------------
def _collect_step(venv, act):
    venv.step_async(act)
    s2, r, d, tr, _ = venv.step_wait()
    S2 = s2.astype(np.float32)
    R = r.reshape(venv.num_envs, 1).astype(np.float32)
    D = d.astype(np.float32).reshape(venv.num_envs, 1)
    T = tr.astype(np.float32).reshape(venv.num_envs, 1)
    return S2, R, D, T

def _eval_one_episode(venv, actor, steps_per_env, env_opts, seed, device):
    # Avaliação limpa: sem ruído, ator em eval(), 1 episódio
    venv.reset_async(seed=seed, options=env_opts)
    s, _ = venv.reset_wait()
    S = s.astype(np.float32)
    actor.eval()
    ret = 0.0
    for _ in range(steps_per_env):
        with torch.no_grad():
            a = actor(torch.as_tensor(S, dtype=torch.float32, device=device))["action"]
        a_np = a.detach().cpu().numpy().astype(np.float32).reshape(venv.num_envs, -1)
        S2, R, D, T = _collect_step(venv, a_np)
        ret += float(np.mean(R))
        S = S2
        if np.all(np.logical_or(D, T)):
            break
    actor.train()
    return ret

# ---------------- train ----------------
def train(actor_arch, model_arch, critics_arch, hp_cls, start_time, start_soc, config_path, model_name, model_weights, seed: Optional[int] = None):
    cfg = json.load(open(config_path, "r"))
    hp = hp_cls(cfg)
    rl = hp.reinforcement_learning

    # Seed: usa argumento se veio; caso contrário, rl.seed do JSON
    if seed is None:
        seed = int(getattr(rl, "seed", 42))
    set_seed_everywhere(seed)

    params = json.load(open("data/parameters.json", "r"))
    start_time = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")

    save_dir = Path(f"saves/{model_name.replace(' ', '_')}")
    save_dir.mkdir(parents=True, exist_ok=True)

    venv, env_opts = make_vec_envs(rl, params, start_time, start_soc)
    venv.reset_async(seed=seed, options=env_opts)
    s, _ = venv.reset_wait()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NF = int(venv.single_observation_space.shape[0])
    NA = int(venv.single_action_space.shape[0])
    S = s.astype(np.float32)

    actor = make_actor_min(venv, hp, actor_arch, model_arch, model_weights)
    actor_targ = deepcopy(actor).to(device)
    # Professor IL fixo para BC
    actor_il_ref = deepcopy(actor).to(device)
    for p in actor_il_ref.parameters():
        p.requires_grad_(False)
    actor_il_ref.eval()

    q, q_targ, soft_update = make_critics_min(venv, hp, critics_arch)

    # Otimizadores conforme optimizer.type + weight_decay
    opt_cls = OPTIMIZERS.get(str(getattr(rl, "opt_type", "adam")).lower(), torch.optim.Adam)
    actor_opt = opt_cls(actor.parameters(), lr=float(rl.lr_actor), weight_decay=float(rl.weight_decay))
    q_opt = opt_cls(q.parameters(), lr=float(rl.lr_critic), weight_decay=float(rl.weight_decay))

    # steps_per_env pelo MAIOR horizonte
    max_h = float(np.max([o["horizon_hours"] for o in env_opts]))
    steps_per_env = int(round(max_h * 60.0 / float(rl.timestep)))

    # <<< MODIFICADO >>>
    # Usa "buffer_size" do JSON (ex: 1e6) em vez de calcular um valor relativo pequeno
    capacity = int(getattr(rl, "buffer_size", 1_000_000)) 
    buffer = ReplayBuffer(obs_shape=(NF,), act_shape=(NA,), capacity=capacity, device=device, dtype=np.float32)

    def make_ckpt(ep, tag, ep_ret):
        return {
            "episode": ep, "tag": tag, "episode_return": float(ep_ret),
            "actor_state": actor.state_dict(), "critic_state": q.state_dict(),
            "actor_targ_state": actor_targ.state_dict(), "critic_targ_state": q_targ.state_dict(),
            "actor_opt": actor_opt.state_dict(), "critic_opt": q_opt.state_dict(),
            "cfg": cfg, "rl": rl.__dict__, "obs_dim": NF, "act_dim": NA
        }

    # ---------------- avaliação inicial (episódio -1) ----------------
    S_eval = S.copy()
    initial_ret = 0.0
    actor.eval()
    for _ in range(steps_per_env):
        with torch.no_grad():
            a_eval = actor(torch.as_tensor(S_eval, dtype=torch.float32, device=device))["action"]
        a_eval_np = a_eval.detach().cpu().numpy().astype(np.float32).reshape(venv.num_envs, -1)
        S_eval2, R_eval, D_eval, T_eval = _collect_step(venv, a_eval_np)
        initial_ret += float(np.mean(R_eval))
        S_eval = S_eval2
        if np.all(np.logical_or(D_eval, T_eval)):
            break
    actor.train()

    metrics_path = save_dir / "best_metrics.json"
    metrics = {
        "history": [{"episode": -1, "return": float(initial_ret)}],
        "eval_history": [],
        "best_return": float(initial_ret),
        "best_episode": -1
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics, f)

    # Reset após avaliação
    venv.reset_async(seed=seed, options=env_opts)
    s, _ = venv.reset_wait()
    S = s.astype(np.float32)

    # ---------------- pretrain dos críticos (opcional) ----------------
    pretrain_steps = int(getattr(rl, "critic_pretrain_steps", 1000))
    last_loss_c = None

    if pretrain_steps > 0:
        pbar = tqdm(range((pretrain_steps + venv.num_envs - 1) // venv.num_envs),
                    desc="Critic pretrain", unit="step", dynamic_ncols=True)
        remaining = pretrain_steps
        while remaining > 0:
            with torch.no_grad():
                a = actor(torch.as_tensor(S, dtype=torch.float32, device=device))["action"]
                noise_std = float(getattr(rl, "pretrain_noise", 0.02))
                if noise_std > 0:
                    a = (a + torch.randn_like(a) * noise_std).clamp_(-1, 1)
            a_np = a.detach().cpu().numpy().astype(np.float32).reshape(venv.num_envs, -1)
            S2, R, D, T = _collect_step(venv, a_np)
            buffer.add_batch(S, a_np, R, S2, D, T)
            S = S2
            remaining -= venv.num_envs

            if len(buffer) >= rl.batch_size:
                b = buffer.sample(rl.batch_size)
                s_b = b["obs"].to(device).float()
                a_b = b["act"].to(device).float()
                r_b = b["rew"].to(device).float()
                s2_b = b["nobs"].to(device).float()
                d_b = b["done"].to(device).float()
                t_b = b["timeout"].to(device).float() if ("timeout" in b) else torch.zeros_like(d_b)

                with torch.no_grad():
                    a2 = actor_targ(s2_b)["action"]
                    # Durante pretrain, use pn inicial/clamp inicial
                    pn0 = float(getattr(rl, "policy_noise", 0.2))
                    nc0 = float(getattr(rl, "noise_clip", 0.5))
                    noise = (torch.randn_like(a2) * pn0).clamp_(-nc0, nc0)
                    a2 = (a2 + noise).clamp_(-1.0, 1.0)
                    q1_t, q2_t = q_targ(s2_b, a2)
                    not_terminal = (1.0 - d_b) * (1.0 - t_b)
                    y = r_b + rl.gamma * not_terminal * torch.min(q1_t, q2_t)

                q1, q2 = q(s_b, a_b)
                loss_c = critic_loss(q1, q2, y, rl)
                q_opt.zero_grad(set_to_none=True)
                loss_c.backward()
                torch.nn.utils.clip_grad_norm_(q.parameters(), float(rl.max_grad_norm))
                q_opt.step()
                last_loss_c = float(loss_c.detach())
                soft_update(q, q_targ, rl.tau)
                soft_update(actor, actor_targ, rl.tau)
            pbar.update(1)
        pbar.close()

    # ---------------- treino principal ----------------
    rew_avg = deque(maxlen=1000)
    updates = 0
    loss_a_val = None
    global_step = 0
    ep_bar = tqdm(range(rl.n_episodes), desc="Episodes", unit="ep", dynamic_ncols=True)
    warmup_eps = int(getattr(rl, "warmup_episodes", 0))
    eval_every = int(getattr(rl, "eval_every", 0))
    last_eval_ret = None

    try:
        for ep in ep_bar:
            # Anneal de ruídos / BC por episódio
            anneal_total = int(getattr(rl, "anneal_episodes", rl.n_episodes))
            en0 = float(getattr(rl, "exploration_noise", 0.1))
            en1 = float(getattr(rl, "exploration_noise_final", en0))
            pn0 = float(getattr(rl, "policy_noise", 0.2))
            pn1 = float(getattr(rl, "policy_noise_final", pn0))
            nc0 = float(getattr(rl, "noise_clip", 0.5))
            nc1 = float(getattr(rl, "noise_clip_final", nc0))
            lb0 = float(getattr(rl, "lambda_bc", 0.0))
            lb1 = float(getattr(rl, "lambda_bc_final", lb0))

            en_cur = lin_anneal(en0, en1, ep, anneal_total)
            pn_cur = lin_anneal(pn0, pn1, ep, anneal_total)
            nc_cur = lin_anneal(nc0, nc1, ep, anneal_total)
            lb_cur = lin_anneal(lb0, lb1, ep, anneal_total)

            venv.reset_async(seed=seed, options=env_opts)
            s, _ = venv.reset_wait()
            S = s.astype(np.float32)
            ep_return = 0.0

            for step in range(steps_per_env):
                with torch.no_grad():
                    a = actor(torch.as_tensor(S, dtype=torch.float32, device=device))["action"]
                    if en_cur > 0:
                        a = (a + torch.randn_like(a) * en_cur).clamp_(-1, 1)
                a_np = a.detach().cpu().numpy().astype(np.float32).reshape(venv.num_envs, -1)

                S2, R, D, T = _collect_step(venv, a_np)
                ep_return += float(np.mean(R))
                rew_avg.append(float(np.mean(R)))
                buffer.add_batch(S, a_np, R, S2, D, T)
                S = S2
                global_step += venv.num_envs

                # Atualizações
                if (global_step % getattr(rl, "update_every", 1) == 0) and (len(buffer) >= rl.batch_size):
                    n_upd = min(int(getattr(rl, "utd", 1)), int(getattr(rl, "max_updates_per_step", 1)))
                    for _ in range(n_upd):
                        b = buffer.sample(rl.batch_size)
                        s_b = b["obs"].to(device).float()
                        a_b = b["act"].to(device).float()
                        r_b = b["rew"].to(device).float()
                        s2_b = b["nobs"].to(device).float()
                        d_b = b["done"].to(device).float()
                        t_b = b["timeout"].to(device).float() if ("timeout" in b) else torch.zeros_like(d_b)

                        # ------ Critic update ------
                        with torch.no_grad():
                            a2 = actor_targ(s2_b)["action"]
                            if pn_cur > 0.0:
                                noise = (torch.randn_like(a2) * pn_cur).clamp_(-nc_cur, nc_cur)
                                a2 = (a2 + noise).clamp_(-1.0, 1.0)
                            q1_t, q2_t = q_targ(s2_b, a2)
                            not_terminal = (1.0 - d_b) * (1.0 - t_b)
                            y = r_b + rl.gamma * not_terminal * torch.min(q1_t, q2_t)

                        q1, q2 = q(s_b, a_b)
                        loss_c = critic_loss(q1, q2, y, rl)
                        q_opt.zero_grad(set_to_none=True)
                        loss_c.backward()
                        torch.nn.utils.clip_grad_norm_(q.parameters(), float(rl.max_grad_norm))
                        q_opt.step()
                        last_loss_c = float(loss_c.detach())

                        # ------ Policy (actor) update ------
                        actor_update_enabled = (ep >= warmup_eps)
                        if actor_update_enabled and ((updates + 1) % rl.policy_delay == 0):
                            a_pred = actor(s_b)["action"]
                            q1_pi, _ = q(s_b, a_pred)
                            loss_a = -q1_pi.mean()

                            # BC leve com referência IL congelada
                            if lb_cur > 0.0:
                                with torch.no_grad():
                                    a_il = actor_il_ref(s_b)["action"]
                                loss_a = loss_a + lb_cur * F.mse_loss(a_pred, a_il)

                            actor_opt.zero_grad(set_to_none=True)
                            loss_a.backward()
                            torch.nn.utils.clip_grad_norm_(actor.parameters(), float(rl.max_grad_norm))
                            actor_opt.step()
                            loss_a_val = float(loss_a.detach())

                            soft_update(actor, actor_targ, rl.tau)
                            soft_update(q, q_targ, rl.tau)

                        updates += 1

                # Logs no tqdm (inclui parâmetros variáveis)
                if (step % 10) == 0:
                    avg_r = (sum(rew_avg) / len(rew_avg)) if rew_avg else None
                    ep_bar.set_postfix(
                        buf=len(buffer),
                        lc=last_loss_c,
                        la=loss_a_val,
                        ar=avg_r,
                        ret=ep_return,
                        en=round(en_cur, 4),
                        pn=round(pn_cur, 4),
                        nc=round(nc_cur, 4),
                        lb=round(lb_cur, 4),
                        eval=(None if last_eval_ret is None else round(last_eval_ret, 3)),
                    )

                if np.all(np.logical_or(D, T)):
                    break

            # ---------------- métricas/ckpts por episódio ----------------
            try:
                with open(metrics_path, "r") as f:
                    metrics = json.load(f)
            except Exception:
                metrics = {"history": [], "eval_history": [], "best_return": float("-inf"), "best_episode": None}

            # <<< MODIFICADO >>>
            # Salva o log completo do episódio (com todos os dados do tqdm)
            avg_r_final = (sum(rew_avg) / len(rew_avg)) if rew_avg else None
            
            episode_log = {
                "episode": int(ep),
                "return": float(ep_return),
                "avg_reward": avg_r_final,
                "last_actor_loss": loss_a_val,
                "last_critic_loss": last_loss_c,
                "buffer_size": len(buffer),
                "explor_noise": round(en_cur, 4),
                "policy_noise": round(pn_cur, 4),
                "noise_clip": round(nc_cur, 4),
                "lambda_bc": round(lb_cur, 4),
                "last_eval_return": (None if last_eval_ret is None else round(last_eval_ret, 3))
            }
            metrics["history"].append(episode_log)
            # <<< FIM DA MODIFICAÇÃO >>>


            if float(ep_return) > float(metrics.get("best_return", float("-inf"))):
                metrics["best_return"] = float(ep_return)
                metrics["best_episode"] = int(ep)

            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=2) # Adicionado indent=2 para legibilidade

            # Save last
            torch.save(make_ckpt(ep, "last", ep_return), save_dir / "last.pt")

            # Save best (considera -1 via best_return)
            current_best = float(metrics["best_return"])
            if float(ep_return) >= current_best:
                torch.save(make_ckpt(ep, "best", ep_return), save_dir / "best_rl.pt")

            # Avaliação periódica limpa
            if eval_every > 0 and ((ep + 1) % eval_every == 0):
                eval_ret = _eval_one_episode(venv, actor, steps_per_env, env_opts, seed, device)
                last_eval_ret = float(eval_ret)
                try:
                    with open(metrics_path, "r") as f:
                        m2 = json.load(f)
                except Exception:
                    m2 = {"history": [], "eval_history": [], "best_return": float("-inf"), "best_episode": None}
                
                # Atualiza o último log de 'history' com o valor de avaliação que acabou de ser calculado
                if m2["history"]:
                    m2["history"][-1]["last_eval_return"] = round(last_eval_ret, 3)

                m2.setdefault("eval_history", []).append({"episode": int(ep), "eval_return": float(eval_ret)})
                with open(metrics_path, "w") as f:
                    json.dump(m2, f, indent=2) # Adicionado indent=2

            # Save periódico
            if getattr(rl, "save_every", 0) and ((ep + 1) % rl.save_every == 0):
                torch.save(make_ckpt(ep, f"ep_{ep+1:05d}", ep_return), save_dir / f"ep_{ep+1:05d}.pt")

    finally:
        try:
            venv.close()
        except Exception:
            pass