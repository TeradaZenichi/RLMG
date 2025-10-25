# models/model1/model.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = [
    "MLP",
    "SquashedGaussianPolicy",
    "QNetwork",
    "TwinQ",
    "make_sac_nets",
]


# ------------------------
# building blocks / utils
# ------------------------

_ACTS = {
    "relu": nn.ReLU,
    "elu": nn.ELU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,   # swish; boa escolha padrão
    "tanh": nn.Tanh,
    "leaky_relu": nn.LeakyReLU,
}

def _activation(name: str) -> nn.Module:
    name = (name or "silu").lower()
    if name not in _ACTS:
        raise ValueError(f"Unknown activation '{name}'. Available: {list(_ACTS)}")
    return _ACTS[name]()

def _init_layers(module: nn.Module, scheme: str = "orthogonal") -> None:
    """
    Pesos padrão p/ RL:
      - 'orthogonal' (ganho=√2) nas camadas lineares + bias zero
    """
    for m in module.modules():
        if isinstance(m, nn.Linear):
            if scheme == "orthogonal":
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain("relu"))
            elif scheme == "xavier":
                nn.init.xavier_uniform_(m.weight)
            else:
                raise ValueError(f"Unknown init scheme '{scheme}'")
            if m.bias is not None:
                nn.init.zeros_(m.bias)


class MLP(nn.Module):
    """
    MLP genérico: [in -> h1 -> ... -> hk] sem camada de saída.
    Use outro Linear fora se precisar mudar dims.
    """
    def __init__(self, in_dim: int, hidden: Sequence[int], activation: str = "silu", init: str = "orthogonal"):
        super().__init__()
        layers: list[nn.Module] = []
        last = in_dim
        act = activation
        for h in hidden:
            layers += [nn.Linear(last, h), _activation(act)]
            last = h
        self.net = nn.Sequential(*layers)
        _init_layers(self, scheme=init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ------------------------------------
# Policy: Squashed Diag-Gaussian MLP
# ------------------------------------

class SquashedGaussianPolicy(nn.Module):
    """
    Política Gaussiana diagonal com squash tanh + reescala para [low, high].
    - Produz ação amostrada, log_prob com correção, e média (determinística).
    - Bounds podem ser definidos depois via set_action_bounds(low, high).
    """
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden: Sequence[int] = (256, 256),
        activation: str = "silu",
        init: str = "orthogonal",
        log_std_bounds: Tuple[float, float] = (-5.0, 2.0),
        action_low: Optional[Iterable[float]] = None,
        action_high: Optional[Iterable[float]] = None,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.backbone = MLP(obs_dim, hidden, activation=activation, init=init)
        last = hidden[-1] if len(hidden) > 0 else obs_dim
        self.mu = nn.Linear(last, act_dim)
        self.log_std = nn.Linear(last, act_dim)
        self.log_std_min, self.log_std_max = float(log_std_bounds[0]), float(log_std_bounds[1])

        # buffers p/ reescala (center + half_range). Default = [-1, 1].
        low = torch.full((act_dim,), -1.0)
        high = torch.full((act_dim,), 1.0)
        if action_low is not None:
            low = torch.as_tensor(list(action_low), dtype=torch.float32)
        if action_high is not None:
            high = torch.as_tensor(list(action_high), dtype=torch.float32)
        center = 0.5 * (high + low)
        half = 0.5 * (high - low)
        self.register_buffer("_act_center", center)
        self.register_buffer("_act_half", half)
        self.eps = 1e-6

        _init_layers(self, scheme=init)
        # saída da média com init menor (±0.01) ajuda a começar "perto do zero"
        nn.init.uniform_(self.mu.weight, -1e-2, 1e-2)
        nn.init.uniform_(self.mu.bias, -1e-2, 1e-2)
        nn.init.uniform_(self.log_std.weight, -1e-2, 1e-2)
        nn.init.constant_(self.log_std.bias, -0.5)  # std inicial ~ exp(-0.5) ≈ 0.61

    @torch.no_grad()
    def set_action_bounds(self, low: Iterable[float], high: Iterable[float]) -> None:
        low_t = torch.as_tensor(list(low), dtype=torch.float32, device=self._act_center.device)
        high_t = torch.as_tensor(list(high), dtype=torch.float32, device=self._act_center.device)
        self._act_center.copy_(0.5 * (high_t + low_t))
        self._act_half.copy_(0.5 * (high_t - low_t))

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retorna (mu, std) ANTES do tanh. Use .sample(...) para ação/log_prob.
        """
        h = self.backbone(obs)
        mu = self.mu(h)
        log_std = self.log_std(h)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        return mu, std

    def _squash(self, u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Aplica tanh e reescala. Retorna:
          - a: ação em [low, high]
          - log_abs_det_jac: soma dos logs |det J| p/ correção do log_prob
        """
        y = torch.tanh(u)  # (-1,1)
        # log|d/d u tanh(u)| = log(1 - tanh(u)^2)
        log_det_tanh = torch.log(1.0 - y.pow(2) + self.eps).sum(dim=-1)
        # reescala: a = center + half * y  →  log|det| soma log(half)
        log_det_scale = torch.log(self._act_half.abs() + self.eps).sum()
        a = self._act_center + self._act_half * y
        return a, (log_det_tanh + log_det_scale)

    def sample(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Amostra (ou usa média) e retorna: (action, log_prob, mean_action)
        - log_prob inclui correção de tanh + escala.
        """
        mu, std = self.forward(obs)
        if deterministic:
            u = mu  # sem ruído
        else:
            dist = torch.distributions.Normal(mu, std)
            u = dist.rsample()  # reparametrização
        a, log_abs_det = self._squash(u)

        # log-prob no espaço não squash
        dist = torch.distributions.Normal(mu, std)
        log_prob = dist.log_prob(u).sum(dim=-1) - log_abs_det  # subtrai log|det J|
        a_mean, _ = self._squash(mu)
        return a, log_prob, a_mean


# --------------------
# Critics (Q-networks)
# --------------------

class QNetwork(nn.Module):
    """
    Q(s,a) como MLP em concat [obs, action].
    """
    def __init__(self, obs_dim: int, act_dim: int, hidden: Sequence[int] = (256, 256),
                 activation: str = "silu", init: str = "orthogonal"):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.backbone = MLP(obs_dim + act_dim, hidden, activation=activation, init=init)
        last = hidden[-1] if len(hidden) > 0 else (obs_dim + act_dim)
        self.q = nn.Linear(last, 1)
        _init_layers(self, scheme=init)
        nn.init.uniform_(self.q.weight, -1e-2, 1e-2)
        nn.init.zeros_(self.q.bias)

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, act], dim=-1)
        h = self.backbone(x)
        return self.q(h).squeeze(-1)


class TwinQ(nn.Module):
    """
    Dois críticos independentes (SAC).
    """
    def __init__(self, obs_dim: int, act_dim: int, hidden: Sequence[int] = (256, 256),
                 activation: str = "silu", init: str = "orthogonal"):
        super().__init__()
        self.q1 = QNetwork(obs_dim, act_dim, hidden, activation, init)
        self.q2 = QNetwork(obs_dim, act_dim, hidden, activation, init)

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.q1(obs, act), self.q2(obs, act)


# -------------------------
# convenience constructor
# -------------------------

@dataclass
class SACNets:
    actor: SquashedGaussianPolicy
    critic: TwinQ

def make_sac_nets(
    obs_dim: int,
    act_dim: int,
    *,
    hidden: Sequence[int] = (256, 256),
    activation: str = "silu",
    init: str = "orthogonal",
    log_std_bounds: Tuple[float, float] = (-5.0, 2.0),
    action_low: Optional[Iterable[float]] = None,
    action_high: Optional[Iterable[float]] = None,
) -> SACNets:
    actor = SquashedGaussianPolicy(
        obs_dim, act_dim, hidden=hidden, activation=activation, init=init,
        log_std_bounds=log_std_bounds, action_low=action_low, action_high=action_high
    )
    critic = TwinQ(obs_dim, act_dim, hidden=hidden, activation=activation, init=init)
    return SACNets(actor=actor, critic=critic)
