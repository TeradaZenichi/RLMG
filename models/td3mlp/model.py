# models/td3mlp/model.py
from typing import List, Dict, Any, Optional
from models import Hyperparameters
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


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
        return self.q_head(z)
    

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
