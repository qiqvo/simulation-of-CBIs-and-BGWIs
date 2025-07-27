import numpy as np
from scipy.special import gamma
from scipy.stats import poisson

from branching_processes_simulation.random_variable.poisson import Poisson
from branching_processes_simulation.random_variable.random_variable import (
    RandomVariable,
)


class ZeroTruncatedPoisson(RandomVariable):
    _interval_a = 1
    _interval_b = +np.inf

    def __init__(self, rate: float) -> None:
        assert 0 < rate
        self.rate = rate
        self._p = Poisson(rate)

    def characteristic_function(self, t: np.complex64) -> np.complex64:
        return (self._p.characteristic_function(t) - self._p.pdf(0)) / (
            1 - self._p.pdf(0)
        )

    def laplace_transform(self, t: np.float64) -> np.float64:
        return (self._p.laplace_transform(t) - self._p.pdf(0)) / (1 - self._p.pdf(0))

    def generating_function(self, s: np.complex64) -> np.complex64:
        return (self._p.generating_function(s) - self._p.pdf(0)) / (1 - self._p.pdf(0))

    def pdf(self, x: np.float64) -> np.float64:
        return self.rate**x * np.exp(-self.rate) / gamma(x + 1)

    def cdf(self, x: np.float64) -> np.float64:
        return poisson.cdf(x, self.rate)

    def mean(self) -> np.float64:
        return self.rate / (1 - np.exp(-self.rate))

    def variance(self) -> np.float64:
        return self.mean() * (1 + self.rate - self.mean())

    def sample(self, N: int, option="cdf", **kwargs) -> np.ndarray[float]:
        if option == "cdf":
            res = poisson.ppf(self.rng.uniform(self.cdf(0), 1, size=N), self.rate)
        elif option == "poisson":
            u = self.rng.uniform(0, 1, size=N)
            u = -np.log(1 - u * (1 - np.exp(-self.rate)))
            res = 1 + self.rng.poisson(self.rate - u, N)
        return res.astype(np.int64)
