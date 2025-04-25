from typing import Callable
import numpy as np
from scipy.special import gamma

from branching_processes_simulation.random_variable.random_variable import RandomVariable


class Poisson(RandomVariable):
    _interval_a = 0
    _interval_b = +np.inf

    def __init__(self, rate: float) -> None:
        assert 0 < rate
        self.rate = rate
        self._rv = self.rng.poisson

    def characteristic_function(self, t: np.complex64) -> np.complex64:
        return np.exp(self.rate * (np.exp(1j * t) - 1))

    def laplace_transform(self, t: np.float64) -> np.float64:
        return np.exp(self.rate * (np.exp(-t) - 1))
    
    def generating_function(self, s: np.complex64) -> np.complex64:
        return np.exp(self.rate * (s - 1))

    def pdf(self, x: np.float64) -> np.float64:
        return self.rate**x * np.exp(-self.rate) / gamma(x + 1)
    
    def cdf(self, x: np.float64) -> np.float64:
        raise NotImplementedError()

    def mean(self) -> np.float64:
        return self.rate

    def variance(self) -> np.float64:
        return self.rate

    def sample(self, N: int) -> np.ndarray[float]:
        return self._rv(self.rate, N)
