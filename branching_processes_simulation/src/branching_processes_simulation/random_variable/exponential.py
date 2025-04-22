
from typing import Callable
import numpy as np
import scipy
from branching_processes_simulation.random_variable.random_variable import RandomVariable


class Exponential(RandomVariable):
    _interval_a = 0
    _interval_b = +np.inf

    def __init__(self, rate: float) -> None:
        assert 0 < rate
        self.rate = rate
        self._rv = self.rng.exponential

    def characteristic_function(self, t: np.complex64) -> np.complex64:
        return 1 / (1 - 1j * t / self.rate)

    def laplace_transform(self, t: np.float64) -> np.float64:
        return 1 / (1 + t / self.rate)

    def pdf(self, x: np.float64) -> np.float64:
        return self.rate * np.exp(-self.rate * x)

    def cdf(self, x: np.float64) -> np.float64:
        return 1 - np.exp(-self.rate * x)

    def mean(self) -> np.float64:
        return 1/self.rate

    def variance(self) -> np.float64:
        return 1/self.rate**2

    def sample(self, N: int) -> np.ndarray[float]:
        return self._rv(1 / self.rate, N)

    def laplace_transform_kth_derivative_at_x(self, k: int, t: np.float64) -> np.float64:
        return self.laplace_transform_kth_derivative(k)(t)

    def laplace_transform_kth_derivative(self, k: int) -> Callable:
        return lambda x: (-1)**k * scipy.special.factorial(k) * (1 + x / self.rate)**(-k-1) * (self.rate)**(-k)
