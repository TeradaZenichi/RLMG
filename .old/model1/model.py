from typing import List, Optional, Dict, Any
import torch
import torch.nn as nn


# ---------------- Hyperparameters ----------------

class Hyperparameters:
    """Mapeia o dicionário de config para atributos com defaults estáveis."""
    def __init__(self, cfg: Dict[str, Any]):
        cfg = dict(cfg or {})
        # Arquitetura
        self.hidden_dims: List[int] = list(cfg.get("hidden_dims", [128, 128]))
        self.dropout: float = float(cfg.get("dropout", 0.0))

        # (mantemos campos de treino para padronização entre modelos)
        self.optimizer: Dict[str, Any] = dict(cfg.get("optimizer", {"type": "adam", "lr": 1e-3, "weight_decay": 0.0}))
        self.loss: Dict[str, Any]      = dict(cfg.get("loss", {"type": "mse"}))
        self.train: Dict[str, Any]     = dict(cfg.get("train", {"batch_size": 256, "epochs": 30}))

        # Guarda cfg bruto para logging/repro
        self.raw: Dict[str, Any] = cfg


# ---------------- Blocos MLP ----------------

class _MLPBlock(nn.Module):
    """Linear -> GELU -> Dropout"""
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.act = nn.GELU()
        self.dp = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dp(self.act(self.fc(x)))


# ---------------- Modelo ----------------

class MLPmodel(nn.Module):
    """
    Input:  state [B, F]
    Output: {"logit": pré-tanh, "action": tanh(logit) em [-1, 1]}
    """
    def __init__(self, cfg: Dict[str, Any], input_dim: int):
        super().__init__()
        self.hp = Hyperparameters(cfg)

        dims = [input_dim] + list(self.hp.hidden_dims)
        layers = []
        for i in range(len(dims) - 1):
            layers.append(_MLPBlock(dims[i], dims[i + 1], dropout=self.hp.dropout))
        self.backbone = nn.Sequential(*layers)

        # Cabeça escalar + Tanh
        self.head = nn.Linear(dims[-1], 1)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.tanh = nn.Tanh()

    def forward(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        z = self.backbone(state)
        logit = self.head(z).squeeze(-1)      # [B]
        action = self.tanh(logit)              # [-1, 1]
        return {"logit": logit, "action": action}
