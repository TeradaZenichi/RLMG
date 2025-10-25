# -*- coding: utf-8 -*-
# ReplayBuffer + HistoryWindow (somente classes do buffer)
from __future__ import annotations
from typing import Optional, Dict
import numpy as np
import torch


class ReplayBuffer:
    """
    ReplayBuffer com escrita vetorizada (add_batch) e amostragem aleatória.
    Armazena: obs, act, rew, nobs, done, timeout.
    """
    def __init__(
        self,
        obs_shape,
        act_shape,
        capacity: int = 1_000_000,
        device: Optional[torch.device] = None,
        dtype=np.float32,
    ):
        self.capacity = int(capacity)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.dtype = dtype

        self.F = int(obs_shape[0]) if isinstance(obs_shape, (tuple, list)) else int(obs_shape)
        self.A = int(act_shape[0]) if isinstance(act_shape, (tuple, list)) else int(act_shape)

        self.obs     = np.zeros((self.capacity, self.F), dtype=self.dtype)
        self.nobs    = np.zeros((self.capacity, self.F), dtype=self.dtype)
        self.act     = np.zeros((self.capacity, self.A), dtype=self.dtype)
        self.rew     = np.zeros((self.capacity, 1), dtype=self.dtype)
        self.done    = np.zeros((self.capacity, 1), dtype=np.float32)
        self.timeout = np.zeros((self.capacity, 1), dtype=np.float32)

        self.ptr = 0
        self.size = 0

    def __len__(self) -> int:
        return int(self.size)

    # ---------- escrita em anel (ring buffer) ----------
    def _assign_ring(self, arr: np.ndarray, start: int, data: np.ndarray) -> None:
        n = data.shape[0]
        cap = self.capacity
        end = start + n
        if end <= cap:
            arr[start:end] = data
        else:
            first = cap - start
            arr[start:cap] = data[:first]
            arr[0:end - cap] = data[first:]

    # ---------- API de inserção ----------
    def add_batch(
        self,
        s: np.ndarray,
        a: np.ndarray,
        r: np.ndarray,
        s2: np.ndarray,
        d: np.ndarray,
        timeout: Optional[np.ndarray] = None,
    ) -> None:
        s  = s.astype(self.dtype, copy=False)
        a  = a.astype(self.dtype, copy=False)
        r  = r.astype(self.dtype, copy=False)
        s2 = s2.astype(self.dtype, copy=False)
        d  = d.astype(np.float32, copy=False)

        if timeout is None:
            timeout = np.zeros_like(d, dtype=np.float32)
        else:
            timeout = timeout.astype(np.float32, copy=False)

        n = int(s.shape[0])
        i = self.ptr

        self._assign_ring(self.obs,     i, s)
        self._assign_ring(self.act,     i, a)
        self._assign_ring(self.rew,     i, r)
        self._assign_ring(self.nobs,    i, s2)
        self._assign_ring(self.done,    i, d)
        self._assign_ring(self.timeout, i, timeout)

        self.ptr = (self.ptr + n) % self.capacity
        self.size = min(self.capacity, self.size + n)

    def add(self, s, a, r, s2, d, timeout=None) -> None:
        self.add_batch(s, a, r, s2, d, timeout)

    def push(self, s, a, r, s2, d, timeout=None) -> None:
        self.add_batch(s, a, r, s2, d, timeout)

    # ---------- amostragem ----------
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        idx = np.random.randint(0, self.size, size=int(batch_size))
        to_t = lambda x: torch.as_tensor(x[idx], device=self.device)
        return {
            "obs":     to_t(self.obs),
            "act":     to_t(self.act),
            "rew":     to_t(self.rew),
            "nobs":    to_t(self.nobs),
            "done":    to_t(self.done),
            "timeout": to_t(self.timeout),
        }


class HistoryWindow:
    """
    Empilha histórico temporal K por ambiente e entrega estado no formato desejado:
      - K=1  -> flat:   [nenvs, F]
      - K>1  -> flat:   [nenvs, K*F]  (adequado para MLPs)
    Útil para TD3/SAC e afins quando queremos dar contexto temporal sem RNN.
    """
    def __init__(
        self,
        nenvs: int,
        feat_dim: int,
        K: int,
        padding: str = "edge",
        dtype: np.dtype = np.float32,
    ):
        self.nenvs = int(nenvs)
        self.F = int(feat_dim)
        self.K = int(max(1, K))
        self.padding = str(padding).lower()
        self.dtype = dtype
        self.buf = np.zeros((self.nenvs, self.K, self.F), dtype=self.dtype)

    @property
    def flat_dim(self) -> int:
        """Dimensão observação empilhada (K*F)."""
        return self.K * self.F

    def reset_with(self, obs_batch: np.ndarray) -> None:
        """Inicializa janela com o primeiro frame (padding estável no início do episódio)."""
        obs = np.asarray(obs_batch, dtype=self.dtype)
        assert obs.shape == (self.nenvs, self.F), f"esperado ({self.nenvs},{self.F}), veio {obs.shape}"
        if self.K == 1:
            self.buf[:, 0, :] = obs
        else:
            if self.padding == "zero":
                self.buf[:] = 0.0
                self.buf[:, -1, :] = obs
            else:  # 'edge' -> repete primeiro frame
                self.buf[:] = obs[:, None, :]

    def stacked(self) -> np.ndarray:
        """Estado atual empilhado (flat)."""
        return self.buf[:, 0, :] if self.K == 1 else self.buf.reshape(self.nenvs, self.K * self.F)

    def push(self, next_obs: np.ndarray) -> None:
        """Insere próximo frame na cauda (shift para a esquerda)."""
        nxt = np.asarray(next_obs, dtype=self.dtype)
        if self.K == 1:
            self.buf[:, 0, :] = nxt
            return
        self.buf[:, :-1, :] = self.buf[:, 1:, :]
        self.buf[:, -1, :] = nxt

    def push_and_stacked(self, next_obs: np.ndarray) -> np.ndarray:
        """Atalho: push() e retorna o estado já empilhado (flat)."""
        self.push(next_obs)
        return self.stacked()

    def clear_env(self, i: int) -> None:
        """Opcional: limpa o histórico do env i (útil em resets parciais)."""
        self.buf[i, :, :] = 0.0

    def set_env_frame(self, i: int, frame: np.ndarray) -> None:
        """Opcional: força o último frame do env i (por exemplo, após reset assíncrono)."""
        f = np.asarray(frame, dtype=self.dtype).reshape(1, self.F)
        if self.K == 1:
            self.buf[i, 0, :] = f
        else:
            self.buf[i, -1, :] = f
