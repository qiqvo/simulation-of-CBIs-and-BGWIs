from collections.abc import Iterable
import numpy as np
from scipy.integrate import quad

from branching_processes_simulation.random_variable.positive_stable import (
    PositiveStable,
)
from branching_processes_simulation.random_variable.random_variable import (
    RandomVariable,
)
from branching_processes_simulation.utils import parallel_integrate_upper_limits


class Vau(RandomVariable):
    _interval_a = 0
    _interval_b = np.pi

    def __init__(self, alpha: float) -> None:
        super().__init__()
        self.alpha = alpha

    def pdf(self, x: np.float64) -> np.float64:
        res = self.alpha / np.pi / PositiveStable.a_shifted(self.alpha, x)
        return res

    def cdf(self, x: np.float64) -> np.float64:
        if isinstance(x, Iterable):
            return parallel_integrate_upper_limits(self.pdf, 0, x)
        else:
            return quad(self.pdf, 0, x)[0]

    def sample(self, N, **kwargs):
        return self.sample_from_cdf(N, True, approximation="linear", **kwargs)

    def characteristic_function(self, t):
        return super().characteristic_function(t)

    def laplace_transform(self, t):
        return super().laplace_transform(t)

    def mean(self) -> np.float64:
        return super().mean()

    def variance(self) -> np.float64:
        return super().variance()
