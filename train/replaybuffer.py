# train/replaybuffer.py
import numpy as np
import torch
from typing import Optional

class ReplayBuffer:
    """
    Minimal vectorized ReplayBuffer.
    Stores flat observations (already stacked by the env), actions in [-1,1],
    rewards, next observations, done (terminal) and truncated (time limit).
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
        self.rew     = np.zeros((self.capacity, 1),      dtype=self.dtype)
        self.done    = np.zeros((self.capacity, 1),      dtype=np.float32)  # 1.0 = terminal
        self.trunc   = np.zeros((self.capacity, 1),      dtype=np.float32)  # 1.0 = time-limit

        self._idx = 0
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def add_batch(self, S, A, R, S2, D, T):
        """
        Vectorized insert of a batch coming from AsyncVectorEnv.
        Shapes:
          S, S2 : [N, F]
          A     : [N, A]
          R     : [N] or [N,1]
          D, T  : [N] or [N,1] (float {0,1})
        """
        S  = np.asarray(S,  dtype=self.dtype)
        S2 = np.asarray(S2, dtype=self.dtype)
        A  = np.asarray(A,  dtype=self.dtype)
        R  = np.asarray(R,  dtype=self.dtype).reshape(-1, 1)
        D  = np.asarray(D,  dtype=np.float32).reshape(-1, 1)
        T  = np.asarray(T,  dtype=np.float32).reshape(-1, 1)

        N = S.shape[0]
        assert S.shape  == (N, self.F) and S2.shape == (N, self.F)
        assert A.shape  == (N, self.A)

        # Circular write with wrap-around
        end = self._idx + N
        if end <= self.capacity:
            sl = slice(self._idx, end)
            self.obs[sl], self.nobs[sl], self.act[sl], self.rew[sl], self.done[sl], self.trunc[sl] = S, S2, A, R, D, T
        else:
            first = self.capacity - self._idx
            sl1 = slice(self._idx, self.capacity)
            sl2 = slice(0, end - self.capacity)
            self.obs[sl1],  self.nobs[sl1],  self.act[sl1],  self.rew[sl1],  self.done[sl1],  self.trunc[sl1]  = S[:first],  S2[:first],  A[:first],  R[:first],  D[:first],  T[:first]
            self.obs[sl2],  self.nobs[sl2],  self.act[sl2],  self.rew[sl2],  self.done[sl2],  self.trunc[sl2]  = S[first:],  S2[first:],  A[first:],  R[first:],  D[first:],  T[first:]

        self._idx = end % self.capacity
        self._size = min(self.capacity, self._size + N)

    def sample(self, batch_size: int):
        """
        Uniform sampling. Returns torch tensors on the target device.
        """
        assert self._size > 0, "Buffer is empty."
        idx = np.random.randint(0, self._size, size=int(batch_size))
        to_t = lambda x: torch.as_tensor(x[idx], device=self.device)
        return {
            "obs":   to_t(self.obs),
            "act":   to_t(self.act),
            "rew":   to_t(self.rew),
            "nobs":  to_t(self.nobs),
            "done":  to_t(self.done),
            "trunc": to_t(self.trunc),
            "idx":   idx,  # handy for prioritized or debugging
        }
