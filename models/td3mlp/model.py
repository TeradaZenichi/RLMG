# models/td3mlp/model.py
from typing import List, Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ======================= CONTEXTOS DE TREINO =======================

class ImitationLearning:
    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        cfg = dict(cfg or {})
        opt = dict(cfg.get("optimizer", {}))
        los = dict(cfg.get("loss", {}))
        trn = dict(cfg.get("train", {}))

        self.horizon       = int(cfg.get("horizon", 24))
        self.timestep      = float(cfg.get("timestep", 5.0))

        self.opt_type      = str(opt.get("type", "adam")).lower()
        self.lr            = float(opt.get("lr", 5e-4))
        self.weight_decay  = float(opt.get("weight_decay", 0.0))

        self.loss_type     = str(los.get("type", "mse")).lower()   # "mse" | "huber"
        self.huber_delta   = float(los.get("delta", 1.0))

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
    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        cfg = dict(cfg or {})
        opt = dict(cfg.get("optimizer", {}))
        trn = dict(cfg.get("train", {}))

        # Ambiente
        self.horizon       = int(cfg.get("horizon", 24))
        self.timestep      = float(cfg.get("timestep", 5.0))
        self.nenvs         = int(cfg.get("nenvs", 1))
        self.ndays         = int(cfg.get("ndays", 7))
        self.stride_hours  = int(cfg.get("stride_hours", 24))
        self.n_episodes    = int(trn.get("n_episodes", 1000))

        # Otimizadores
        self.opt_type      = str(opt.get("type", "adam")).lower()
        base_lr            = float(opt.get("lr", 1e-3))
        self.lr_actor      = float(trn.get("lr_actor", base_lr))
        self.lr_critic     = float(trn.get("lr_critic", base_lr))
        self.weight_decay  = float(opt.get("weight_decay", 0.0))

        # TD3 específicos
        self.gamma         = float(trn.get("gamma", 0.99))
        self.tau           = float(trn.get("tau", 0.005))
        self.batch_size    = int(trn.get("batch_size", 256))
        self.start_steps   = int(trn.get("start_steps", 5000))
        self.update_after  = int(trn.get("update_after", 1000))
        self.update_every  = int(trn.get("update_every", 1))
        self.updates_per_step = int(trn.get("updates_per_step", 1))

        self.exploration_noise = float(trn.get("exploration_noise", 0.1))  # σ para coleta
        self.policy_noise   = float(trn.get("policy_noise", 0.2))          # σ no alvo
        self.noise_clip     = float(trn.get("noise_clip", 0.5))            # clip no alvo
        self.policy_delay   = int(trn.get("policy_delay", 2))              # atraso da política

        self.raw = cfg


# ======================= HYPERPARAMETERS (arquitetura + contextos) =======================

class Hyperparameters:
    def __init__(self, cfg: Dict[str, Any]):
        cfg = dict(cfg or {})
        obs = dict(cfg.get("obs", {}).get("history", {}))

        self.hidden_dims: List[int] = list(cfg.get("hidden_dims", [256, 256]))
        self.dropout: float         = float(cfg.get("dropout", 0.0))
        self.raw: Dict[str, Any]    = cfg

        # História/entrada
        self.window_size: int       = int(obs.get("window", 1))
        self.obs_mode: str          = str(obs.get("mode", "flat")).lower()

        # Contextos
        self.imitation_learning     = ImitationLearning(cfg.get("imitation_learning", {}))
        self.reinforcement_learning = ReinforcementLearning(cfg.get("reinforcement_learning", {}))


# ======================= IL: MLP DETERMINÍSTICO =======================

class MLPmodel(nn.Module):
    """ Cabeça determinística para IL (ação em [-1,1]). """
    def __init__(self, hp: Hyperparameters, input_dim: int):
        super().__init__()
        self.hp = hp
        dims = [input_dim] + self.hp.hidden_dims
        layers = []
        for i in range(len(dims)-1):
            fc = nn.Linear(dims[i], dims[i+1])
            nn.init.xavier_uniform_(fc.weight); nn.init.zeros_(fc.bias)
            layers += [fc, nn.GELU(), nn.Dropout(self.hp.dropout)]
        self.mlp = nn.Sequential(*layers)

        last = dims[-1] if self.hp.hidden_dims else input_dim
        self.head = nn.Linear(last, 1)
        nn.init.xavier_uniform_(self.head.weight); nn.init.zeros_(self.head.bias)
        self.tanh = nn.Tanh()

    def forward(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        z = self.mlp(state)
        logit = self.head(z).squeeze(-1)
        action = self.tanh(logit)
        return {"logit": logit, "action": action}


# ======================= TD3: ACTOR DETERMINÍSTICO =======================

class DeterministicTanhActor(nn.Module):
    """ Ator determinístico π(s) ∈ [-1,1]^A para TD3 (MLP). """
    def __init__(self, hp: Hyperparameters, input_dim: int, action_dim: int = 1):
        super().__init__()
        self.hp = hp
        dims = [input_dim] + self.hp.hidden_dims
        layers = []
        for i in range(len(dims)-1):
            fc = nn.Linear(dims[i], dims[i+1])
            nn.init.xavier_uniform_(fc.weight); nn.init.zeros_(fc.bias)
            layers += [fc, nn.GELU(), nn.Dropout(self.hp.dropout)]
        self.mlp = nn.Sequential(*layers)

        last = dims[-1] if self.hp.hidden_dims else input_dim
        self.mu_head = nn.Linear(last, action_dim)
        nn.init.xavier_uniform_(self.mu_head.weight); nn.init.zeros_(self.mu_head.bias)

    def forward(self, state: torch.Tensor, deterministic: bool = True) -> Dict[str, torch.Tensor]:
        z = self.mlp(state)
        mu = torch.tanh(self.mu_head(z))
        return {"action": mu}


# ======================= TD3/SAC: CRITICS (compartilhados) =======================

class QCritic(nn.Module):
    """ Q(s,a) com MLP; entrada: concat(state, action). """
    def __init__(self, hp: Hyperparameters, state_dim: int, action_dim: int = 1):
        super().__init__()
        self.hp = hp
        in_dim = state_dim + action_dim
        dims = [in_dim] + self.hp.hidden_dims
        layers = []
        for i in range(len(dims)-1):
            fc = nn.Linear(dims[i], dims[i+1])
            nn.init.xavier_uniform_(fc.weight); nn.init.zeros_(fc.bias)
            layers += [fc, nn.GELU(), nn.Dropout(self.hp.dropout)]
        self.mlp = nn.Sequential(*layers)

        last = dims[-1] if self.hp.hidden_dims else in_dim
        self.q_head = nn.Linear(last, 1)
        nn.init.xavier_uniform_(self.q_head.weight); nn.init.zeros_(self.q_head.bias)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        z = self.mlp(x)
        return self.q_head(z).squeeze(-1)


class TwinQCritic(nn.Module):
    """ Dois críticos independentes (Q1, Q2) para Double Q. """
    def __init__(self, hp: Hyperparameters, state_dim: int, action_dim: int = 1):
        super().__init__()
        self.q1 = QCritic(hp, state_dim, action_dim)
        self.q2 = QCritic(hp, state_dim, action_dim)

    def forward(self, state: torch.Tensor, action: torch.Tensor):
        return self.q1(state, action), self.q2(state, action)


# ======================= EXPORTS =======================

__all__ = [
    "ImitationLearning", "ReinforcementLearning",
    "Hyperparameters",
    "MLPmodel",
    "DeterministicTanhActor",
    "QCritic", "TwinQCritic",
]
