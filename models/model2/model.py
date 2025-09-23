from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = [
    "MLP",
    "ConstrainedSquashedGaussianPolicy",
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


# --------------------------------------------------------------
# Policy: Squashed Diag-Gaussian with STATE-DEPENDENT bounds
# --------------------------------------------------------------
class ConstrainedSquashedGaussianPolicy(nn.Module):
    """
    Política Gaussiana diagonal com squash tanh e ESCALA DINÂMICA por estado
    para garantir ação factível por construção.

    A escala dinâmica [lo_eff(obs), hi_eff(obs)] é calculada usando:
      - SoC (obs[-1]) e parâmetros do BESS (E_bess_kwh, eta_ch/dis, soc_min/max)
      - dt_minutes (para converter energia -> potência)
      - Limites de hardware (±P_max)
      - Limites de rede (grid_export_kw_min <= grid_p <= grid_import_kw_max)
      - Net load (load - pv) reconstruído a partir de obs[load_frac, pv_frac]
        e dos Pmax (Load_Pmax_kw, PV_Pmax_kw)

    Assim, a ação da política já respeita SoC + hardware + limites de rede,
    eliminando mismatch entre treino e execução no ambiente.

    IMPORTANTE: chame set_env_context(cfg) uma vez antes de treinar/avaliar.
    """
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden: Sequence[int] = (256, 256),
        activation: str = "silu",
        init: str = "orthogonal",
        log_std_bounds: Tuple[float, float] = (-5.0, 2.0),
        safety_temp: float = 1.0,
    ):
        super().__init__()
        assert act_dim == 1, "Este ambiente usa ação 1D (kW)."
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.backbone = MLP(obs_dim, hidden, activation=activation, init=init)
        last = hidden[-1] if len(hidden) > 0 else obs_dim
        self.mu = nn.Linear(last, act_dim)
        self.log_std = nn.Linear(last, act_dim)
        self.log_std_min, self.log_std_max = float(log_std_bounds[0]), float(log_std_bounds[1])

        # context/env buffers (preenchidos via set_env_context)
        # escalas
        self.register_buffer("_PV_Pmax_kw", torch.tensor(1.0))
        self.register_buffer("_Load_Pmax_kw", torch.tensor(1.0))
        # BESS
        self.register_buffer("_E_bess_kwh", torch.tensor(1.0))
        self.register_buffer("_eta_ch", torch.tensor(1.0))
        self.register_buffer("_eta_dis", torch.tensor(1.0))
        self.register_buffer("_soc_min", torch.tensor(0.0))
        self.register_buffer("_soc_max", torch.tensor(1.0))
        # hardware
        self.register_buffer("_P_ch_max_kw", torch.tensor(1.0))
        self.register_buffer("_P_dis_max_kw", torch.tensor(1.0))
        # tempo
        self.register_buffer("_dt_h", torch.tensor(0.25))  # default 15 min
        # rede
        self.register_buffer("_grid_imp_max", torch.tensor(float("inf")))
        self.register_buffer("_grid_exp_min", torch.tensor(float("-inf")))

        # indices na observação (conforme environment)
        self.PV_FRAC_IDX = 0
        self.LOAD_FRAC_IDX = 1
        self.SOC_IDX = -1

        self.safety_temp = float(safety_temp)  # opcional, p/ controlar "dureza" do squash
        self.eps = 1e-6

        _init_layers(self, scheme=init)
        # inits suaves para estabilidade no começo
        nn.init.uniform_(self.mu.weight, -1e-2, 1e-2)
        nn.init.uniform_(self.mu.bias, -1e-2, 1e-2)
        nn.init.uniform_(self.log_std.weight, -1e-2, 1e-2)
        nn.init.constant_(self.log_std.bias, -0.5)  # std inicial ~ exp(-0.5) ≈ 0.61

    @torch.no_grad()
    def set_env_context(self, cfg_like) -> None:
        """
        Passe EnergyEnvConfig (ou objeto compatível) para definir o contexto.
        """
        device = self.mu.weight.device
        def as_t(x): return torch.as_tensor(float(x), dtype=torch.float32, device=device)

        self._PV_Pmax_kw.copy_(as_t(getattr(cfg_like, "PV_Pmax_kw", 1.0)))
        self._Load_Pmax_kw.copy_(as_t(getattr(cfg_like, "Load_Pmax_kw", 1.0)))
        self._E_bess_kwh.copy_(as_t(getattr(cfg_like, "E_bess_kwh", 1.0)))
        self._eta_ch.copy_(as_t(getattr(cfg_like, "eta_ch", 1.0)))
        self._eta_dis.copy_(as_t(getattr(cfg_like, "eta_dis", 1.0)))
        self._soc_min.copy_(as_t(getattr(cfg_like, "soc_min", 0.0)))
        self._soc_max.copy_(as_t(getattr(cfg_like, "soc_max", 1.0)))
        self._P_ch_max_kw.copy_(as_t(getattr(cfg_like, "P_ch_max_kw", getattr(cfg_like, "Pmax_kw", 1.0))))
        self._P_dis_max_kw.copy_(as_t(getattr(cfg_like, "P_dis_max_kw", getattr(cfg_like, "Pmax_kw", 1.0))))
        dt_minutes = int(getattr(cfg_like, "dt_minutes", 15))
        self._dt_h.copy_(as_t(max(dt_minutes / 60.0, 1e-9)))
        self._grid_imp_max.copy_(as_t(getattr(cfg_like, "grid_import_kw_max", float("inf"))))
        self._grid_exp_min.copy_(as_t(getattr(cfg_like, "grid_export_kw_min", float("-inf"))))

    # -------- bounds dinâmicos --------
    def _compute_dynamic_bounds(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calcula [lo_eff, hi_eff] por amostra, respeitando:
          - SoC (energia disponível e "headroom")
          - Hardware
          - Grid (via net load = load - pv)
        """
        # extrai features (B x 1)
        pv_frac  = obs[..., self.PV_FRAC_IDX:self.PV_FRAC_IDX+1]
        ld_frac  = obs[..., self.LOAD_FRAC_IDX:self.LOAD_FRAC_IDX+1]
        soc      = obs[..., self.SOC_IDX:self.SOC_IDX+1].clamp(0.0, 1.0)

        # reconstrói kW
        pv_kw = pv_frac * self._PV_Pmax_kw
        ld_kw = ld_frac * self._Load_Pmax_kw
        net   = ld_kw - pv_kw  # grid = net + p_eff

        # física (limites de potência AC por passo via SoC)
        E_up = (self._soc_max - soc).clamp(min=0.0) * self._E_bess_kwh                # kWh
        E_dn = (soc - self._soc_min).clamp(min=0.0) * self._E_bess_kwh                # kWh
        p_eff_max_physical = (E_up / (self._eta_ch + self.eps)) / self._dt_h          # kW
        p_eff_min_physical = - (self._eta_dis * E_dn) / self._dt_h                    # kW

        # hardware
        p_eff_max_hw = self._P_ch_max_kw
        p_eff_min_hw = -self._P_dis_max_kw

        # grid (faixa de p_eff que mantém grid dentro dos limites)
        p_eff_min_grid = self._grid_exp_min - net
        p_eff_max_grid = self._grid_imp_max - net

        # interseção
        lo_eff = torch.maximum(p_eff_min_physical, p_eff_min_hw)
        lo_eff = torch.maximum(lo_eff, p_eff_min_grid)
        hi_eff = torch.minimum(p_eff_max_physical, p_eff_max_hw)
        hi_eff = torch.minimum(hi_eff, p_eff_max_grid)

        # trata casos degenerados: se hi<lo, relaxa os limites de rede (mantém SoC+HW)
        infeasible = (hi_eff < lo_eff)
        if torch.any(infeasible):
            lo_ph_hw = torch.maximum(p_eff_min_physical, p_eff_min_hw)
            hi_ph_hw = torch.minimum(p_eff_max_physical, p_eff_max_hw)
            lo_eff = torch.where(infeasible, lo_ph_hw, lo_eff)
            hi_eff = torch.where(infeasible, hi_ph_hw, hi_eff)

        # evita half=0 (faixa vazia) → dá epsilon de respiro
        half = (hi_eff - lo_eff) * 0.5
        tiny = (half.abs() < 1e-6)
        hi_eff = torch.where(tiny, hi_eff + 5e-4, hi_eff)
        lo_eff = torch.where(tiny, lo_eff - 5e-4, lo_eff)

        return lo_eff, hi_eff

    # -------- política base --------
    def _forward_stats(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(obs)
        mu = self.mu(h)
        log_std = self.log_std(h).clamp(self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        return mu, std

    def _squash_and_scale(self, u: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Aplica tanh com temperatura e reescala para [lo, hi] (dependente do estado).
        Retorna:
          - a: ação kW
          - log_abs_det_jac: soma dos logs |det J| p/ correção do log_prob
        """
        # tanh com "temperatura" -> y in (-1, 1)
        if self.safety_temp != 1.0:
            u = u / max(self.safety_temp, 1e-6)
        y = torch.tanh(u)

        # Jacobiano do tanh
        log_det_tanh = torch.log(1.0 - y.pow(2) + self.eps).sum(dim=-1)

        # escala dinâmica
        center = 0.5 * (hi + lo)
        half = 0.5 * (hi - lo).clamp_min(1e-6)
        a = center + half * y  # (B,1)

        # log|det escala|: soma dos logs(half)
        log_det_scale = torch.log(half + self.eps).sum(dim=-1)
        return a, (log_det_tanh + log_det_scale)

    def sample(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Amostra (ou usa média) e retorna: (action, log_prob, mean_action).
        A ação já está em kW e dentro dos limites dinâmicos do estado.
        """
        lo, hi = self._compute_dynamic_bounds(obs)
        mu, std = self._forward_stats(obs)

        if deterministic:
            u = mu
        else:
            dist = torch.distributions.Normal(mu, std)
            u = dist.rsample()

        a, log_abs_det = self._squash_and_scale(u, lo, hi)

        # log-prob no espaço não squash
        dist = torch.distributions.Normal(mu, std)
        log_prob = dist.log_prob(u).sum(dim=-1) - log_abs_det

        # ação média (determinística)
        a_mean, _ = self._squash_and_scale(mu, lo, hi)
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
    actor: ConstrainedSquashedGaussianPolicy
    critic: TwinQ

def make_sac_nets(
    obs_dim: int,
    act_dim: int,
    *,
    hidden: Sequence[int] = (256, 256),
    activation: str = "silu",
    init: str = "orthogonal",
    log_std_bounds: Tuple[float, float] = (-5.0, 2.0),
    safety_temp: float = 1.0,
) -> SACNets:
    """
    Constrói ator/critico do model2.
    Após construir, CHAME `actor.set_env_context(cfg)` para passar o contexto.
    """
    actor = ConstrainedSquashedGaussianPolicy(
        obs_dim, act_dim, hidden=hidden, activation=activation, init=init,
        log_std_bounds=log_std_bounds, safety_temp=safety_temp
    )
    critic = TwinQ(obs_dim, act_dim, hidden=hidden, activation=activation, init=init)
    return SACNets(actor=actor, critic=critic)
