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
        self.warmup_random = bool(cfg.get("warmup_random", False))

       



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