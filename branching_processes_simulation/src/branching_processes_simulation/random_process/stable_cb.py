from typing import List
import numpy as np

from branching_processes_simulation.random_process.cb import CriticalCB
from branching_processes_simulation.random_variable.tau import Tau


class StableCB(CriticalCB):
    def __init__(self, alpha: np.float64, c: np.float64) -> None:
        assert 0 < alpha <= 1 and c > 0
        super().__init__(lambda t: c * t ** (1 + alpha))
        self.alpha = alpha
        self.c = c
        self._tau = Tau(alpha)

    def characteristic_function(
        self, t: np.complex64, time: np.float64, z: np.float64
    ) -> np.complex64:
        return self.laplace_transform(-1j * t, time, z)

    def laplace_transform(
        self, t: np.float64, time: float, z: np.float64
    ) -> np.float64:
        return np.exp(
            -t
            * z
            / np.power(
                1 + self.alpha * self.c * time * np.abs(t) ** self.alpha, 1 / self.alpha
            )
        )

    def variance(self, time: float, z: np.float64) -> np.float64:
        if self.alpha < 1:
            return np.infty
        else:
            # TODO: check:
            return 2 * self.c * time

    def sample(
        self, N: int, time: np.float64, z: List[np.float64], **kwargs
    ) -> np.ndarray[np.ndarray[float]]:
        k = (self.alpha * self.c * time) ** (1 / self.alpha)
        m = len(z)

        # if z[i] == 0:
        #     continue
        s = self.rng.poisson(np.array(z) / k, size=(N, m))  # (N, m)
        if (np.sum(s)) == 0:
            return np.zeros((m, N), np.float64)

        X = self._tau.sample(np.sum(s), **kwargs) * k
        b = np.cumulative_sum(s[s > 0][:-1], include_initial=True)
        (np.zeros((m, N), np.float64)).T[s > 0] = np.add.reduceat(X, b)

        return np.zeros((m, N), np.float64)
