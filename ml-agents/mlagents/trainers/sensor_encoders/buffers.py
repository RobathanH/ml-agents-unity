from typing import Optional, Tuple

import numpy as np


class RingBuffer:
    def __init__(self, capacity: int, obs_dim: int):
        self.capacity = int(capacity)
        self.obs_dim = int(obs_dim)
        self.data = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self.size = 0
        self.ptr = 0

    def push(self, obs: np.ndarray) -> None:
        arr = obs.astype(np.float32).reshape(-1)
        if arr.shape[0] != self.obs_dim:
            # Truncate or pad if needed to match expected dimension
            if arr.shape[0] > self.obs_dim:
                arr = arr[: self.obs_dim]
            else:
                pad = np.zeros((self.obs_dim - arr.shape[0],), dtype=np.float32)
                arr = np.concatenate([arr, pad], axis=0)
        self.data[self.ptr] = arr
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def can_sample(self, batch_size: int) -> bool:
        return self.size >= batch_size

    def sample(self, batch_size: int) -> np.ndarray:
        idx = np.random.randint(0, self.size, size=(batch_size,))
        return self.data[idx]

    def stats(self) -> Tuple[int, int]:
        return self.size, self.capacity


