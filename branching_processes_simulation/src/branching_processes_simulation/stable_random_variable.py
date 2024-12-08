from typing import Any, Callable
import numpy as np

from branching_processes_simulation.random_variable import RandomVariable


# TODO: unstable near alpha=1: 
class PositiveStableRandomVariable(RandomVariable):
    # alpha < 1
    def __init__(self, alpha: float, d: float=1) -> None:
        assert 0 < alpha <= 1 and d > 0
        self.alpha = alpha
        self.d = d

    def characteristic_function(self, t: np.complex64) -> np.complex64:
        return np.exp(- self.d * np.power(t, self.alpha))

    def laplace_transform(self, t: np.float64) -> np.float64:
        return np.real(self.characteristic_function(t))

    def pdf(self, x: np.float64) -> np.float64:
        return None # unknown

    def cdf(self, x: np.float64) -> np.float64:
        return None # unknown

    def mean(self) -> np.float64:
        if self.alpha < 1:
            return np.infty
        elif self.alpha == 1:
            return self.d
        else:
            return None 

    def variance(self) -> np.float64:
        if self.alpha < 1:
            return np.infty
        else:
            return 0

    def _a(self, theta: np.ndarray[float]):
        c2 = np.sin(self.alpha * theta)
        c2 = np.power(c2, self.alpha / (1 - self.alpha))
        return np.sin((1 - self.alpha) * theta) * c2 / np.power(np.sin(theta), 1/(1 - self.alpha))

    def sample(self, N: int) -> np.ndarray[float]:
        theta = self.rng.uniform(0, 1, N)
        w = -np.log(self.rng.uniform(0, 1, N))
        return np.power(self._a(theta) / w, (1 - self.alpha) / self.alpha) * (self.d**(1/self.alpha))
    
    def sample_function(self, N: int, theta: Callable[..., Any]) -> np.ndarray[float]:
        s = self.sample(N)
        return theta(s)
