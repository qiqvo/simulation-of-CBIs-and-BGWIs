from typing import Any, Callable
import numpy as np

from branching_processes_simulation.constant_variable import ConstantVariable
from branching_processes_simulation.continuous_space_process.fejer_de_la_vallee_poussin_random_variable import FejerDeLaValleePoussinRandomVariable
from branching_processes_simulation.random_variable import RandomVariable


# TODO: unstable near alpha=1: 
class PositiveStableRandomVariable(RandomVariable):
    def __new__(cls, alpha: float, d: float=1, *args, **kwargs):
        if alpha == 1:
            return ConstantVariable(d)
        return super().__new__(cls)
    
    # alpha < 1
    def __init__(self, alpha: float, d: float=1) -> None:
        assert 0 < alpha <= 1 and d > 0
        self.alpha = alpha
        self.d = d
        self._fvp = FejerDeLaValleePoussinRandomVariable()

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

    def sample(self, N: int, option='CMS') -> np.ndarray[float]:
        alpha = self.alpha
        if option=='CMS':
            theta = self.rng.uniform(0, 1, N)
            w = -np.log(self.rng.uniform(0, 1, N))
            return np.power(self._a(theta) / w, (1 - alpha) / alpha) * (self.d**(1/alpha))
        elif option == 'polya':
            v = np.abs(self._fvp.sample(N))
            u1, u2 = self.rng.uniform(0, 1, (2, N))
            s = np.log(np.max(self.alpha / (u1 * u2), 1/u1)) ** (1 / alpha)
            return v / s * (self.d**(1/alpha))
    
    def sample_function(self, N: int, theta: Callable[..., Any]) -> np.ndarray[float]:
        s = self.sample(N)
        return theta(s)
