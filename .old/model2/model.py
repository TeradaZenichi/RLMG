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
    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        cfg = dict(cfg or {})
        opt = dict(cfg.get("optimizer", {}))
        trn = dict(cfg.get("train", {}))

        # Treino (ambiente / episódios)
        self.horizon       = int(cfg.get("horizon", 24))
        self.timestep      = float(cfg.get("timestep", 5.0))
        self.nenvs         = int(cfg.get("nenvs", 1))
        self.ndays         = int(cfg.get("ndays", 7))
        self.stride_hours  = int(cfg.get("stride_hours", 24))   # pode estar obsoleto, mas mantido
        self.n_episodes    = int(trn.get("n_episodes", 1000))

        # Otimizadores
        self.opt_type      = str(opt.get("type", "adam")).lower()
        base_lr            = float(opt.get("lr", 1e-3))
        self.lr_actor      = float(trn.get("lr_actor", base_lr))
        self.lr_critic     = float(trn.get("lr_critic", base_lr))
        self.lr_alpha      = float(trn.get("lr_alpha", 3e-4))
        self.weight_decay  = float(opt.get("weight_decay", 0.0))

        # SAC
        self.gamma          = float(trn.get("gamma", 0.995))
        self.tau            = float(trn.get("tau", 0.005))
        self.auto_alpha     = bool(trn.get("auto_alpha", True))
        self.target_entropy = float(trn.get("target_entropy", -1.0))  # ação 1D
        self.alpha_init     = float(trn.get("alpha_init", 0.2))

        # Passo de treino
        self.batch_size       = int(trn.get("batch_size", 256))
        self.buffer_capacity  = int(trn.get("buffer_capacity", 500_000))
        self.total_steps      = int(trn.get("total_steps", 30_000))
        self.warmup_steps     = int(trn.get("warmup_steps", 5_000))
        self.updates_per_step = int(trn.get("updates_per_step", 1))

        # Estabilidade / exploração
        self.max_grad_norm = float(trn.get("max_grad_norm", 0.0)) or None
        self.n_step        = int(trn.get("n_step", 1))
        self.reward_scale  = float(trn.get("reward_scale", 1.0))

        # Warm-start do ator com IL
        self.log_std_init   = float(trn.get("log_std_init", -0.5))
        self.bc_weight      = float(trn.get("bc_weight", 0.0))      # legado (ainda suportado)
        self.bc_decay_steps = int(trn.get("bc_decay_steps", 0))

        # Avaliação/salvamento
        self.eval_every_episodes = int(trn.get("eval_every_episodes", 0))
        self.eval_episodes       = int(trn.get("eval_episodes", 0))
        self.save_every_episodes = int(trn.get("save_every_episodes", 0))

        # ---------------------- NOVOS PARÂMETROS ----------------------

        # Regularização BC (nova forma, compatível com JSON novo)
        # fallback para bc_weight se bc_coef_max não vier no JSON
        self.bc_coef_max    = float(trn.get("bc_coef_max", trn.get("bc_weight", 0.0)))  # NOVO
        self.bc_coef_min    = float(trn.get("bc_coef_min", 0.0))                        # NOVO
        # (bc_decay_steps já mapeado acima)

        # Controle da temperatura (α)
        self.freeze_alpha_steps = int(trn.get("freeze_alpha_steps", 0))  # NOVO
        self.alpha_clamp_min    = float(trn.get("alpha_clamp_min", 1e-3))# NOVO
        self.alpha_clamp_max    = float(trn.get("alpha_clamp_max", 1.0)) # NOVO

        # Delay de atualização do ator/α
        self.policy_delay = int(trn.get("policy_delay", 1))              # NOVO

        # Limites para o log_std do ator
        self.log_std_clamp_min = float(trn.get("log_std_clamp_min", -5.0))  # NOVO
        self.log_std_clamp_max = float(trn.get("log_std_clamp_max",  2.0))  # NOVO

        # --------------------------------------------------------------

        self.seed = int(trn.get("seed", 42))
        self.raw = cfg



# ======================= HYPERPARAMETERS (arquitetura + contextos) =======================

class Hyperparameters:
    def __init__(self, cfg: Dict[str, Any]):
        cfg = dict(cfg or {})
        self.hidden_dims: List[int] = list(cfg.get("hidden_dims", [128, 128]))
        self.dropout: float = float(cfg.get("dropout", 0.0))
        self.log_std_min: float = float(cfg.get("log_std_min", -5.0))
        self.log_std_max: float = float(cfg.get("log_std_max",  2.0))
        self.raw: Dict[str, Any] = cfg

        # contextos de treinamento (instanciados aqui a partir do JSON)
        self.imitation_learning = ImitationLearning(cfg.get("imitation_learning", {}))
        self.reinforcement_learning = ReinforcementLearning(cfg.get("reinforcement_learning", {}))


# ======================= IL: MLP DETERMINÍSTICO =======================

class MLPmodel(nn.Module):
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

        out_in = dims[-1] if self.hp.hidden_dims else input_dim
        self.head = nn.Linear(out_in, 1)
        nn.init.xavier_uniform_(self.head.weight); nn.init.zeros_(self.head.bias)
        self.tanh = nn.Tanh()

    def forward(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        z = self.mlp(state)
        logit = self.head(z).squeeze(-1)
        action = self.tanh(logit)
        return {"logit": logit, "action": action}


# ======================= SAC: ACTOR =======================

class GaussianTanhActor(nn.Module):
    def __init__(self, hp: "Hyperparameters", input_dim: int, action_dim: int = 1):
        super().__init__()
        self.hp = hp

        # MLP tronco
        dims = [input_dim] + self.hp.hidden_dims
        layers = []
        for i in range(len(dims) - 1):
            fc = nn.Linear(dims[i], dims[i + 1])
            nn.init.xavier_uniform_(fc.weight); nn.init.zeros_(fc.bias)
            layers += [fc, nn.GELU(), nn.Dropout(self.hp.dropout)]
        self.mlp = nn.Sequential(*layers)

        last = dims[-1] if self.hp.hidden_dims else input_dim
        self.mu_head      = nn.Linear(last, action_dim)
        self.log_std_head = nn.Linear(last, action_dim)
        nn.init.xavier_uniform_(self.mu_head.weight);      nn.init.zeros_(self.mu_head.bias)
        nn.init.xavier_uniform_(self.log_std_head.weight); nn.init.zeros_(self.log_std_head.bias)

        # Limites de clamp do log_std vindos do JSON novo (com fallback)
        rl = getattr(self.hp, "reinforcement_learning", self.hp)  # fallback se o objeto encapsula direto
        self.log_std_min = float(getattr(rl, "log_std_clamp_min", getattr(self.hp, "log_std_clamp_min", -5.0)))
        self.log_std_max = float(getattr(rl, "log_std_clamp_max", getattr(self.hp, "log_std_clamp_max",  2.0)))

    def forward(self, state: torch.Tensor, deterministic: bool = False) -> Dict[str, torch.Tensor]:
        z = self.mlp(state)
        mu = self.mu_head(z)

        # clamp do log_std com novos limites
        log_std = torch.clamp(self.log_std_head(z), min=self.log_std_min, max=self.log_std_max)
        std = log_std.exp()

        # Amostragem reparametrizada antes do tanh
        if deterministic:
            pre_tanh = mu
        else:
            eps = torch.randn_like(mu)
            pre_tanh = mu + std * eps

        action = torch.tanh(pre_tanh)

        log_prob = None
        if not deterministic:
            # log-prob gaussiana de pre_tanh
            const = math.log(2.0 * math.pi)
            log_prob_gauss = -0.5 * ((pre_tanh - mu) ** 2 / (std * std + 1e-8) + 2.0 * log_std + const)
            log_prob_gauss = log_prob_gauss.sum(dim=-1)

            # termo de correção da transformação tanh, forma estável:
            # log |det d(tanh)/du| = sum(2*(log(2) - u - softplus(-2u)))
            log_det_correction = (2.0 * (math.log(2.0) - pre_tanh - F.softplus(-2.0 * pre_tanh))).sum(dim=-1)

            log_prob = log_prob_gauss - log_det_correction

        return {"mu": mu, "log_std": log_std, "action": action, "log_prob": log_prob}

    @torch.no_grad()
    def init_from_supervised(self, il_model: "MLPmodel", log_std_init: Optional[float] = None):
        # Copia tronco MLP do IL
        il_layers = [m for m in il_model.mlp if isinstance(m, nn.Linear)]
        ac_layers = [m for m in self.mlp     if isinstance(m, nn.Linear)]
        for src, dst in zip(il_layers, ac_layers):
            dst.weight.copy_(src.weight); dst.bias.copy_(src.bias)

        # Copia cabeça mu do IL (se houver)
        if isinstance(il_model.head, nn.Linear):
            self.mu_head.weight.copy_(il_model.head.weight)
            self.mu_head.bias.copy_(il_model.head.bias)

        # Inicializa log_std com valor do JSON (rl.log_std_init) ou argumento
        rl = getattr(self.hp, "reinforcement_learning", self.hp)
        init_val = float(rl.log_std_init if log_std_init is None else log_std_init)
        self.log_std_head.weight.zero_()
        self.log_std_head.bias.fill_(init_val)


# ======================= SAC: CRITICS =======================

class QCritic(nn.Module):
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
    "MLPmodel", "GaussianTanhActor",
    "QCritic", "TwinQCritic",
]
