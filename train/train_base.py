from torch.utils.data import Dataset, DataLoader
from datetime import datetime
from tqdm.auto import tqdm
from pathlib import Path
from opt import Teacher
import torch.nn as nn
import json, random
import pandas as pd
import numpy as np            # <<< NOVO
import torch

# from models.model2.model import Hyperparameters, MLPmodel


# ---------------- reproducibility ----------------
def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------- tiny dataset ----------------
class PairDataset(Dataset):
    def __init__(self, states, actions):
        # states: [N,F] ou [N,K,F] já prontos (se vier [N,K,F] e o modelo for MLP,
        # vamos achatar ANTES de criar o dataset; ver bloco mais abaixo)
        self.x = torch.tensor(states, dtype=torch.float32)
        y = torch.tensor(actions, dtype=torch.float32)
        self.y = y.view(-1) if y.ndim > 1 else y

    def __len__(self): return self.x.shape[0]
    def __getitem__(self, idx): return self.x[idx], self.y[idx]


# ---------------- helpers ----------------
def make_criterion(loss_type: str = "mse", delta: float = 1.0) -> nn.Module:
    loss_type = str(loss_type).lower()
    if loss_type == "huber":
        return nn.HuberLoss(delta=float(delta))
    return nn.MSELoss()

def make_optimizer(model: nn.Module, opt_type="adam", lr=1e-3, weight_decay=0.0, momentum=0.9):
    opt_type = str(opt_type).lower()
    if opt_type == "sgd":
        return torch.optim.SGD(model.parameters(), lr=float(lr), weight_decay=float(weight_decay), momentum=float(momentum))
    return torch.optim.Adam(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))

def build_history(states, actions, cfg):
    S = np.asarray(states, dtype=np.float32)
    y = np.asarray(actions, dtype=np.float32).reshape(-1)
    N, F = S.shape[0], S.shape[1]

    hist_cfg = (cfg.get("obs", {}) or {}).get("history", {}) if isinstance(cfg, dict) else {}
    enabled  = bool(hist_cfg.get("enabled", False))
    if not enabled:
        return S, y  

    K       = int(hist_cfg.get("window", 12))
    stride  = int(hist_cfg.get("stride", 1))
    mode    = str(hist_cfg.get("mode", "flat")).lower()
    padding = str(hist_cfg.get("padding", "edge")).lower()

    pad_left = max(0, K - 1)
    if padding == "zero":
        pad_block = np.zeros((pad_left, F), dtype=np.float32)
    elif padding == "circular" and N >= pad_left:
        pad_block = S[-pad_left:, :]
    else:  
        pad_block = np.repeat(S[:1, :], repeats=pad_left, axis=0)

    S_pad = np.concatenate([pad_block, S], axis=0)  

    X_list, y_list = [], []
    for i in range(0, N, stride):
        w = S_pad[i:i+K, :]
        if w.shape[0] < K: break
        if mode == "sequence":
            X_list.append(w)                 
        else:
            X_list.append(w.reshape(K * F))  
        y_list.append(y[i])

    X = np.stack(X_list, axis=0)
    y = np.asarray(y_list, dtype=np.float32)
    return X, y


def run_epoch(model, loader, criterion, optim=None, device="cpu", desc="train", grad_clip=None):
    training = optim is not None
    model.train(training)
    total, n = 0.0, 0

    pbar = tqdm(loader, desc=desc, leave=False)
    for xb, yb in pbar:
        xb, yb = xb.to(device), yb.to(device)

        with torch.set_grad_enabled(training):
            pred = model(xb)["action"]    # [-1, 1], shape [B]
            loss = criterion(pred, yb)
            if training:
                optim.zero_grad()
                loss.backward()
                if grad_clip and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip))
                optim.step()

        bs = xb.size(0)
        total += loss.item() * bs
        n += bs
        pbar.set_postfix(loss=f"{loss.item():.4f}", avg=f"{(total/max(1,n)):.4f}")

    return total / max(1, n)


# ---------------- main ----------------
def train(start_time: str, start_soc: float,
          model_arch, hp_cls, config_path: str,
          model_name: str):
    set_seed(42)

    # ---- configs ----
    cfg = json.load(open(config_path, "r"))
    hp  = hp_cls(cfg)  
    il  = hp.imitation_learning


    params = json.load(open("data/parameters.json", "r"))

    # ---- teacher data ----
    pv   = pd.read_csv("data/pv_5min_train.csv", index_col=0, parse_dates=True)
    load = pd.read_csv("data/load_5min_train.csv", index_col=0, parse_dates=True)
    start_time = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")

    teacher = Teacher.Teacher(params, pv, load)
    teacher.horizon = il.horizon
    teacher.Δt = il.timestep
    teacher.build(start_time, start_soc)
    teacher.solve()

    states  = teacher.normalized_states   # [N, F] (já normalizados)
    actions = teacher.normalized_actions  # [N] em [-1, 1]

    X, y = build_history(states, actions, cfg)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if X.ndim == 3:                     # [N,K,F]
        X = X.reshape(X.shape[0], -1)   # [N, K*F]
    input_dim = X.shape[1]              # robusto a ambos os casos
    model = model_arch(hp, input_dim=input_dim).to(device)

    batch_size = int(il.batch_size)
    epochs     = int(il.epochs)
    criterion  = make_criterion(il.loss_type, il.huber_delta)
    optimizer  = make_optimizer(model, il.opt_type, il.lr, il.weight_decay)

    N = X.shape[0]
    idx = list(range(N)); random.shuffle(idx)
    k = int((1.0 - float(il.val_split)) * N)
    idx_tr, idx_va = idx[:k], idx[k:]
    sub = lambda arr, ids: [arr[i] for i in ids] if isinstance(arr, list) else arr[ids]

    X_tr, y_tr = sub(X, idx_tr), sub(y, idx_tr)
    X_va, y_va = sub(X, idx_va), sub(y, idx_va)

    ds_tr = PairDataset(X_tr, y_tr)
    ds_va = PairDataset(X_va, y_va)

    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=bool(il.shuffle), num_workers=int(il.num_workers))
    dl_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False,                    num_workers=int(il.num_workers))

    save_dir = Path(f"saves/{model_name}"); save_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = save_dir / "best_il.pt"
    hp_path   = save_dir / "hp.json"

    best_val = float("inf")
    outer = tqdm(range(1, epochs + 1), desc="epochs", leave=True)
    for ep in outer:
        tr = run_epoch(model, dl_tr, criterion, optim=optimizer, device=device,
                       desc=f"train {ep}/{epochs}", grad_clip=il.grad_clip)
        va = run_epoch(model, dl_va, criterion, optim=None, device=device,
                       desc=f"valid {ep}/{epochs}")

        outer.set_postfix(train=f"{tr:.4f}", val=f"{va:.4f}")

        if va < best_val:
            best_val = va
            torch.save({
                "model_state": model.state_dict(),
                "input_dim": int(input_dim),
                "cfg": cfg
            }, best_ckpt)
            with open(hp_path, "w") as f:
                json.dump(cfg, f, indent=2)

    print(f"Best val loss: {best_val:.6f} | saved to {best_ckpt} and {hp_path}")


if __name__ == "__main__":
    train()
