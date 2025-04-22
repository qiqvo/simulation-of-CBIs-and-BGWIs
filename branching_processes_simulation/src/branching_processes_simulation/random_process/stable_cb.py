from typing import Callable, List
import numpy as np
from scipy.stats import poisson

from branching_processes_simulation.random_process.cb import CriticalCB
from branching_processes_simulation.random_variable.tau import Tau


class StableCB(CriticalCB):
    def __init__(self, alpha: np.float64, c: np.float64) -> None:
        assert 0 < alpha <= 1 and c > 0
        super().__init__(lambda t: c * t**(1 + alpha))
        self.alpha = alpha
        self.c = c
        self._xi = Tau(alpha)

    def characteristic_function(self, t: np.complex64, time: np.float64, z: np.float64) -> np.complex64:
        return self.laplace_transform(-1j * t, time, z)

    def laplace_transform(self, t: np.float64, time: float, z: np.float64) -> np.float64:
        return np.exp(-t*z / np.power(1 + self.alpha * self.c * time * np.abs(t)**self.alpha, 1 / self.alpha))

    def variance(self,  time: float, z: np.float64) -> np.float64:
        if self.alpha < 1:
            return np.infty
        else: 
            # TODO: check:
            return 2 * self.c * time

    def sample(self, N: int, time: np.float64, z: List[np.float64], **kwargs) -> np.ndarray[np.ndarray[float]]:
        k = (self.alpha * self.c * time)**(1 / self.alpha)
        m = len(z)
        S = np.zeros((m, N), np.float64)
        for i in range(len(z)):
            s = self.rng.poisson(z[i] / k, size=N)
            
            X = self._xi.sample(np.sum(s), **kwargs)
            b = np.cumulative_sum(s[:-1], include_initial=True)
            S[i, :] = np.add.reduceat(X, b) * k
            # for i in range(N):
            #     s[i] = np.sum(self._xi.sample(s[i], **kwargs)) * k
        return S