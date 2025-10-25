from train.replaybuffer import ReplayBuffer, HistoryWindow
from gymnasium.vector import AsyncVectorEnv
from datetime import datetime, timedelta
from copy import deepcopy
import pandas as pd
import numpy as np
import torch, inspect
import json


from environment import EnergyEnv


def make_vec_envs(rl, params, start_time, start_soc):
    nenvs, ndays, timestep = rl.nenvs, rl.ndays, rl.timestep
    nenvs = min(nenvs, max(1, ndays))
    base = pd.Timestamp(start_time).normalize()
    days = [(base + timedelta(days=i)).strftime("%Y-%m-%d %H:%M:%S") for i in range(ndays)]

    q, r = divmod(ndays, nenvs)                                   # q: dias base por env; r: quantos recebem +1
    sizes = [q + (1 if i < r else 0) for i in range(nenvs)]       # ex.: ndays=8, nenvs=3 -> [3,3,2]
    blocks, j = [], 0
    for ssz in sizes: blocks.append(days[j:j+ssz]); j += ssz

    env_opts = [{"start_time": b[0], "horizon_hours": 24*len(b)} for b in blocks]

    def make_env(i):
        def _thunk(i=i):
            p = json.loads(json.dumps(params))
            return EnergyEnv(p,
                start_time=env_opts[i]["start_time"],
                soc_ini=start_soc,
                horizon_hours=env_opts[i]["horizon_hours"],
                timestep=timestep)
        return _thunk

    venv = AsyncVectorEnv([make_env(i) for i in range(nenvs)])
    return venv, env_opts




def make_actor_min(venv, hp, actor_arch, model_arch, model_weights):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    F = int(venv.single_observation_space.shape[0])   # nº de features
    A = int(venv.single_action_space.shape[0])        # dim da ação
    K = int(getattr(hp, "window_size", 1))            # janela (frames)
    actor_input_dim = K * F                            # ator MLP espera flat
    actor = actor_arch(hp, input_dim=actor_input_dim, action_dim=A).to(device)
    if not model_weights:
        return actor
    
    try:
        ckpt = torch.load(model_weights, map_location=device)
        il_state = ckpt.get("model_state", None) if isinstance(ckpt, dict) else None
        act_sd = actor.state_dict()
        transferable = {k: v for k, v in il_state.items()
                        if k in act_sd and act_sd[k].shape == v.shape}
        act_sd.update(transferable)
        if ("head.weight" in il_state and "mu_head.weight" in act_sd and
            act_sd["mu_head.weight"].shape == il_state["head.weight"].shape):
            act_sd["mu_head.weight"] = il_state["head.weight"]
        if ("head.bias" in il_state and "mu_head.bias" in act_sd and
            act_sd["mu_head.bias"].shape == il_state["head.bias"].shape):
            act_sd["mu_head.bias"] = il_state["head.bias"]
        actor.load_state_dict(act_sd, strict=False)

    except Exception:
        print("")
        pass

    return actor

import torch


def make_critics_min(venv, hp, twincritic_cls):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    F = int(venv.single_observation_space.shape[0])   # nº de features
    A = int(venv.single_action_space.shape[0])        # dim da ação
    K = int(getattr(hp, "window_size", 1))            # janela (frames)
    state_dim = K * F                                  # crítico MLP espera flat

    q      = twincritic_cls(hp, state_dim=state_dim, action_dim=A).to(device)
    q_targ = twincritic_cls(hp, state_dim=state_dim, action_dim=A).to(device)
    q_targ.load_state_dict(q.state_dict())            # hard update inicial

    lr_c = float(hp.reinforcement_learning.lr_critic)
    wd   = float(hp.reinforcement_learning.weight_decay)
    q_opt = torch.optim.Adam(q.parameters(), lr=lr_c, weight_decay=wd)

    def soft_update(src, dst, tau: float):
        with torch.no_grad():
            for p, pt in zip(src.parameters(), dst.parameters()):
                pt.mul_(1.0 - tau).add_(tau * p)

    return q, q_targ, q_opt, soft_update




def train(actor_arch, model_arch, critics_arch, hp_cls, start_time, start_soc, config_path, model_name, model_weights):
    cfg = json.load(open(config_path, "r"))
    hp  = hp_cls(cfg)  
    
    pv   = pd.read_csv("data/pv_5min_train.csv", index_col=0, parse_dates=True)
    load = pd.read_csv("data/load_5min_train.csv", index_col=0, parse_dates=True)
    start_time = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
    params = json.load(open("data/parameters.json", "r"))
    
    # ------------ Parallel Environments ------------
    venv, env_opts = make_vec_envs(hp.reinforcement_learning, params, start_time, start_soc)
    venv.reset_async(seed=None, options=env_opts)
    s, _ = venv.reset_wait()

    
    # ------------- AI Model --------------
    actor = make_actor_min(venv, hp, actor_arch, model_arch, model_weights)
    # supondo que você tenha importado TwinQCritic do mesmo módulo do ator
    q, q_targ, q_opt, soft_update = make_critics_min(venv, hp, twincritic_cls=critics_arch)


    actor_targ = deepcopy(actor)
    actor_targ.load_state_dict(actor.state_dict())
    actor_opt = torch.optim.Adam(actor.parameters(), lr=hp.reinforcement_learning.lr_actor, weight_decay=hp.reinforcement_learning.weight_decay)

    device = next(actor.parameters()).device
    F = int(venv.single_observation_space.shape[0])
    A = int(venv.single_action_space.shape[0])
    K = int(getattr(hp, "window_size", 1))

    hist = HistoryWindow(nenvs=venv.num_envs, feat_dim=F, K=K)
    hist.reset_with(s)
    obs_dim = K * F

    # substituir isso pelo hiperparâmetro que mostra a quantidade de elementos do buffer
    avg_h = float(np.mean([opt["horizon_hours"] for opt in env_opts]))
    steps_per_env = int(round(avg_h * 60.0 / float(hp.reinforcement_learning.timestep)))
    capacity = int(steps_per_env * venv.num_envs * 4)  # margem

    buffer = ReplayBuffer(obs_shape=(obs_dim,), act_shape=(A,), capacity=capacity, device=device, dtype=np.float32)

    # ------------- Loop de coleta inicial (simplificado) --------------
    warm = dict(hp.raw.get("learning_rate", {}))
    warmup_random = bool(warm.get("warmup_random", False))
    start_steps   = int(getattr(hp.reinforcement_learning, "start_steps", 0))
    base_sigma    = float(getattr(hp.reinforcement_learning, "exploration_noise", 0.1))
    sigma         = float(warm.get("warmup_exploration_noise", max(0.02, 0.5 * base_sigma)))
    iters         = (start_steps + venv.num_envs - 1) // venv.num_envs if start_steps > 0 else 0

    for _ in range(iters):
        S = hist.stacked().astype(np.float32)

        if warmup_random:
            a_norm = np.random.uniform(-1.0, 1.0, size=(venv.num_envs, A)).astype(np.float32)
        else:
            with torch.no_grad():
                a_norm = actor(torch.as_tensor(S, dtype=torch.float32, device=device))["action"].cpu().numpy()
            
        venv.step_async(a_norm.astype(np.float32))
        s2, r, term, trunc, _ = venv.step_wait()

        buffer.add_batch(
            S,
            a_norm,
            r.reshape(venv.num_envs, 1).astype(np.float32),
            hist.push_and_stacked(s2).astype(np.float32),
            (term | trunc).astype(np.float32).reshape(venv.num_envs, 1),
            trunc.astype(np.float32).reshape(venv.num_envs, 1),
        )

        if np.all(term | trunc):
            break


    # ------------ Loop principal de interação + treino TD3 ------------
    # TODO(main loop):
    # total_steps_target = hp.reinforcement_learning.n_episodes * steps_per_env * venv.num_envs   # ou outro critério
    # updates = 0
    # for step in range(total_steps_target):
    #     # 1) Coleta uma etapa com a política atual (com ruído de exploração):
    #     S = hist.stacked()
    #     with torch.no_grad():
    #         a_norm = actor(torch.as_tensor(S, dtype=torch.float32, device=device))["action"].cpu().numpy()
    #     # ruído de exploração TD3:
    #     epsilon = np.random.normal(0.0, hp.reinforcement_learning.exploration_noise, size=a_norm.shape).astype(np.float32)
    #     a_norm = np.clip(a_norm + epsilon, -1.0, 1.0)
    #     a_env = scale_action_to_env(a_norm, low=venv.single_action_space.low, high=venv.single_action_space.high)
    #     venv.step_async(a_env)
    #     s2, r, term, trunc, infos = venv.step_wait()
    #     d = (term | trunc).astype(np.float32).reshape(-1, 1)
    #     timeout = trunc.astype(np.float32).reshape(-1, 1)
    #     S2 = hist.push_and_stacked(s2)
    #     rb.add_batch(S.astype(np.float32), a_norm.astype(np.float32), r.reshape(-1,1).astype(np.float32), S2.astype(np.float32), d, timeout)
    #
    #     # 2) A partir de update_after, treinar a cada update_every passos:
    #     if step >= hp.reinforcement_learning.update_after and (step % hp.reinforcement_learning.update_every == 0):
    #         for _ in range(hp.reinforcement_learning.updates_per_step):
    #             batch = rb.sample(hp.reinforcement_learning.batch_size)
    #             obs  = batch["obs"].to(device).float()
    #             act  = batch["act"].to(device).float()
    #             rew  = batch["rew"].to(device).float()
    #             nobs = batch["nobs"].to(device).float()
    #             done = batch["done"].to(device).float()
    #             timeout = batch["timeout"].to(device).float()
    #
    #             # Máscara terminal: se usar time-limit, (1 - done) substituído por (1 - done) ou (1 - done)*(1 - timeout)
    #             not_done = 1.0 - done  # ou: (1.0 - done) * (1.0 - timeout)
    #
    #             # 2.1) Alvo TD3 com smoothing:
    #             with torch.no_grad():
    #                 # ação alvo com ruído (policy smoothing) e clamp em [-1,1]
    #                 a_targ = actor_targ(nobs)["action"]
    #                 noise = torch.randn_like(a_targ) * hp.reinforcement_learning.policy_noise
    #                 noise = torch.clamp(noise, -hp.reinforcement_learning.noise_clip, hp.reinforcement_learning.noise_clip)
    #                 a_targ = torch.clamp(a_targ + noise, -1.0, 1.0)
    #
    #                 # Q alvo dos críticos alvo e min(Q1, Q2)
    #                 q1_t, q2_t = q_targ(nobs, a_targ)
    #                 q_t_min = torch.min(q1_t, q2_t).unsqueeze(-1) if q1_t.ndim == 1 else torch.min(q1_t, q2_t)
    #
    #                 # y = r + gamma * not_done * q_t_min
    #                 y = rew + hp.reinforcement_learning.gamma * not_done * q_t_min
    #
    #             # 2.2) Atualiza críticos (MSE)
    #             q1, q2 = q(obs, act)
    #             q1 = q1.unsqueeze(-1) if q1.ndim == 1 else q1
    #             q2 = q2.unsqueeze(-1) if q2.ndim == 1 else q2
    #             critic_loss = torch.nn.functional.mse_loss(q1, y) + torch.nn.functional.mse_loss(q2, y)
    #             q_opt.zero_grad(set_to_none=True)
    #             critic_loss.backward()
    #             if hp.imitation_learning.grad_clip:  # se quiser reutilizar esse hiper
    #                 torch.nn.utils.clip_grad_norm_(q.parameters(), hp.imitation_learning.grad_clip)
    #             q_opt.step()
    #
    #             # 2.3) Policy delay: atualiza ator a cada N updates
    #             if (updates % hp.reinforcement_learning.policy_delay) == 0:
    #                 # ator maximiza Q1(s, π(s))
    #                 a_pi = actor(obs)["action"]
    #                 actor_loss = -q.q1(obs, a_pi).mean() if hasattr(q, "q1") else -q(obs, a_pi)[0].mean()
    #                 actor_opt.zero_grad(set_to_none=True)
    #                 actor_loss.backward()
    #                 if hp.imitation_learning.grad_clip:
    #                     torch.nn.utils.clip_grad_norm_(actor.parameters(), hp.imitation_learning.grad_clip)
    #                 actor_opt.step()
    #
    #                 # soft update dos alvos
    #                 soft_update(q,      q_targ,  tau=hp.reinforcement_learning.tau)
    #                 soft_update(actor,  actor_targ, tau=hp.reinforcement_learning.tau)
    #
    #             updates += 1
    #
    #     # 3) (Opcional) se todos envs terminaram, pode reconfigurar blocos/episódios ou sair
    #     #    if all_done: break ou reset com novos env_opts
    #
    #     # 4) (Logs/monitoramento) acumular métricas e imprimir/salvar periodicamente:
    #     #    - reward médio recente, perdas (critic/actor), SoC médio, PV usado, energia de rede, etc.
    #
    # # 5) Avaliação (sem ruído) em janelas fixas:
    # #    reset envs -> rodar política determinística a_norm = actor(S)["action"] sem ruído, mapear e executar.
    # #    computar KPIs (custo total, shed/curtail, tarifa média, etc.)
    #
    # # 6) Checkpoints:
    # #    salvar state_dict de actor, q, alvos, otimizadores e hp.raw em f"{model_name}_*.pt"
    # #    se save_best_only: manter o melhor por métrica (ex.: menor custo médio na avaliação)
    #
    # # 7) (Opcional) Amp/mixed precision:
    # #    usar torch.cuda.amp.autocast e GradScaler para etapas de backward dos críticos/ator

 














    a = 1
    