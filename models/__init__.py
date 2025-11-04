from typing import List, Dict, Any, Optional
import torch.nn.functional as F
import math
import torch
import torch.nn as nn


# ======================= CONTEXTOS DE TREINO =======================

class ImitationLearning:
    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        cfg = dict(cfg or {})
        opt = dict(cfg.get("optimizer", {}))
        los = dict(cfg.get("loss", {}))
        trn = dict(cfg.get("train", {}))

        # Otimizador / perda
        self.horizon       = int(cfg.get("horizon", 24))
        self.timestep      = float(cfg.get("timestep", 5)) 

        self.opt_type      = str(opt.get("type", "adam")).lower()
        self.lr            = float(opt.get("lr", 5e-4))
        self.weight_decay  = float(opt.get("weight_decay", 0.0))

        self.loss_type     = str(los.get("type", "mse")).lower()  # "mse" | "huber"
        self.huber_delta   = float(los.get("delta", 1.0))

        # Treino
        self.batch_size    = int(trn.get("batch_size", 128))
        self.epochs        = int(trn.get("epochs", 20))
        self.shuffle       = bool(trn.get("shuffle", True))
        self.val_split     = float(trn.get("val_split", 0.1))
        self.early_stopping_patience = int(trn.get("early_stopping_patience", 0))
        self.save_best_only= bool(trn.get("save_best_only", True))
        self.grad_clip     = float(trn.get("grad_clip", 0.0)) or None
        self.num_workers   = int(trn.get("num_workers", 0))
        self.seed          = int(trn.get("seed", 42))

        self.raw = cfg



class ReinforcementLearning:
    def __init__(self, cfg=None):
        cfg = dict(cfg or {})
        opt = dict(cfg.get("optimizer", {}))
        trn = dict(cfg.get("train", {}))

        # ----- Environment / data collection -----
        self.buffer_size = int(trn.get("buffer_size", 1e6))
        self.horizon = int(cfg.get("horizon", 24))
        self.timestep = float(cfg.get("timestep", 5.0))
        self.nenvs = int(cfg.get("nenvs", 1))
        self.ndays = int(cfg.get("ndays", 7))
        self.stride_hours = int(cfg.get("stride_hours", 24))
        self.n_episodes = int(trn.get("n_episodes", 100))

        # Observation window override if desired
        self.window_size = int(cfg.get("window_size", trn.get("window_size", 1)))

        # ----- Optimizers -----
        self.opt_type = str(opt.get("type", "adam")).lower()
        self.lr_actor = float(trn.get("lr_actor", 1e-4))
        self.lr_critic = float(trn.get("lr_critic", 1e-3))
        self.lr_alpha = float(trn.get("lr_alpha", 3e-4))
        self.weight_decay = float(opt.get("weight_decay", 0.0))

        # ----- Buffer / batching -----
        self.batch_size = int(trn.get("batch_size", 256))

        # Legacy warmup controls
        self.warmup_random = bool(cfg.get("warmup_random", trn.get("warmup_random", False)))
        self.start_steps = int(trn.get("start_steps", 0))
        self.warmup_utd = float(trn.get("warmup_utd", 1))

        # ----- TD3 / DDPG core -----
        self.gamma = float(trn.get("gamma", 0.99))
        self.tau = float(trn.get("tau", 0.005))
        self.policy_delay = int(trn.get("policy_delay", 2))
        self.policy_noise = float(trn.get("policy_noise", 0.1))
        self.noise_clip = float(trn.get("noise_clip", 0.2))
        self.utd = float(trn.get("utd", 1))

        # ----- Scheduler of updates and exploration -----
        self.update_every = int(trn.get("update_every", 1))
        self.max_updates_per_step = int(trn.get("max_updates_per_step", 1))
        self.exploration_noise = float(trn.get("exploration_noise", 0.03))

        # >>>>>>> NOVOS CAMPOS (do JSON atualizado) <<<<<<<
        self.warmup_episodes = int(trn.get("warmup_episodes", 0))
        self.eval_every = int(trn.get("eval_every", 0))
        self.anneal_episodes = int(trn.get("anneal_episodes", self.n_episodes))

        self.exploration_noise_final = float(trn.get("exploration_noise_final", self.exploration_noise))
        self.policy_noise_final = float(trn.get("policy_noise_final", self.policy_noise))
        self.noise_clip_final = float(trn.get("noise_clip_final", self.noise_clip))
        self.lambda_bc_final = float(trn.get("lambda_bc_final", trn.get("lambda_bc", 0.0)))

        # ----- Critic pretrain and behavior regularization -----
        self.critic_pretrain_steps = int(trn.get("critic_pretrain_steps", 10000))
        self.pretrain_noise = float(trn.get("pretrain_noise", 0.02))
        self.lambda_bc = float(trn.get("lambda_bc", 0.001))

        # ----- Numerics and logging -----
        self.max_grad_norm = float(trn.get("max_grad_norm", 10.0))
        self.save_every = int(trn.get("save_every", 50))
        self.seed = int(trn.get("seed", 42))

        # ----- Loss knobs -----
        self.use_huber = bool(trn.get("use_huber", True))
        self.huber_delta = float(trn.get("huber_delta", 1.0))


    def to_dict(self):
        return self.__dict__




       



# ======================= HYPERPARAMETERS (arquitetura + contextos) =======================

class Hyperparameters:
    def __init__(self, cfg: Dict[str, Any]):
        cfg = dict(cfg or {})
        obs = dict(cfg.get("obs", {}).get("history", {}))
        self.hidden_dims: List[int]     = list(cfg.get("hidden_dims", [128, 128]))
        self.dropout: float             = float(cfg.get("dropout", 0.0))
        self.log_std_min: float         = float(cfg.get("log_std_min", -5.0))
        self.log_std_max: float         = float(cfg.get("log_std_max",  2.0))
        self.raw: Dict[str, Any]        = cfg
        self.window_size: int           = int(obs.get("window", 1))
        self.obs_mode                   = str(obs.get("mode", "flat")).lower()


        # contextos de treinamento (instanciados aqui a partir do JSON)
        self.imitation_learning = ImitationLearning(cfg.get("imitation_learning", {}))
        self.reinforcement_learning = ReinforcementLearning(cfg.get("reinforcement_learning", {}))