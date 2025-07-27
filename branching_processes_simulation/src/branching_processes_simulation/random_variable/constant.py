from typing import Callable
import numpy as np

from branching_processes_simulation.random_variable.random_variable import (
    RandomVariable,
)


class Constant(RandomVariable):
    _interval_a = -np.inf
    _interval_b = +np.inf

    def __init__(self, const) -> None:
        self._const = const

    def characteristic_function(self, t: np.complex64) -> np.complex64:
        return np.exp(1j * t * self._const)

    def laplace_transform(self, t: np.float64) -> np.float64:
        return np.exp(-t * self._const)

    def pdf(self, x: np.float64) -> np.float64:
        return np.inf if x == self._const else 0

    def cdf(self, x: np.float64) -> np.float64:
        return 1 if x >= self._const else 0

    def mean(self) -> np.float64:
        return self._const

    def variance(self) -> np.float64:
        return 0

    def sample(self, N: int, **kwargs) -> np.ndarray[float]:
        return np.ones((N)) * self._const

    def sample_function(self, N: int, theta: Callable, **kwargs) -> np.ndarray[float]:
        return np.ones((N)) * theta(self._const)

    def function_expectation(
        self, theta: Callable, N=100, **kwargs
    ) -> np.ndarray[float]:
        return theta(self._const)

    def sample_from_cdf(
        self, N: int, pdf_available=False, **kwargs
    ) -> np.ndarray[float]:
        return self.sample(N)
