import typing
import numpy as np
from scipy.stats import poisson

from branching_processes_simulation.continuous_space_process.cb import CB
from branching_processes_simulation.continuous_space_process.tau import Tau


class StableCB(CB):
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

    def mean(self, time: float, z: np.float64) -> np.float64:
        return z

    # TODO: check:
    def variance(self,  time: float, z: np.float64) -> np.float64:
        if self.alpha < 1:
            return np.infty
        else: 
            return 2 * self.c * time

    def sample_on_time(self, N: int, time: float, z: np.float64, **kwargs) -> np.ndarray[float]:
        k = (self.alpha * self.c * time)**(1 / self.alpha)
        s = poisson.rvs(z / k, size=N)
        for i in range(N):
            s[i] = np.sum(self._xi.sample(s[i], **kwargs)) * k
        return s
    
    def sample_function(self, N: int, time: float, theta: typing.Callable, z: np.float64) -> np.ndarray[float]:    
        # TODO: 
        return None
