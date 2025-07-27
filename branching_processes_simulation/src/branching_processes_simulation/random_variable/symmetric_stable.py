from typing import Any, Callable
import numpy as np
import scipy

from branching_processes_simulation.random_variable.constant import Constant
from branching_processes_simulation.random_variable.fejer_de_la_vallee_poussin_random_variable import (
    FejerDeLaValleePoussin,
)
from branching_processes_simulation.random_variable.stable import Stable


class SymmetricStable(Stable):
    _interval_a = -np.inf
    _interval_b = +np.inf

    def __new__(cls, alpha: float, d: float = 1, *args, **kwargs):
        if alpha == 1:
            return Constant(d)
        return super().__new__(cls)

    def __init__(self, alpha: float, d: float = 1) -> None:
        super().__init__(alpha, 0, d)
        self._fvp = FejerDeLaValleePoussin()

    def laplace_transform(self, t: np.float64) -> np.float64:
        return np.nan

    def characteristic_function(self, t: np.complex64) -> np.complex64:
        return np.exp(-self.d * np.power(np.abs(t), self.alpha))

    def pdf(self, x: np.float64) -> np.float64:
        raise NotImplementedError()

    def cdf(self, x: np.float64) -> np.float64:
        raise NotImplementedError()

    def sample(self, N: int, option="scipy", **kwargs) -> np.ndarray[float]:
        alpha = self.alpha
        if option == "polya":
            v = self._fvp.sample(N)
            u1, u2 = self.rng.uniform(0, 1, (2, N))
            s = np.log(np.maximum(self.alpha / (u1 * u2), 1 / u1)) ** (1 / alpha)
            res = v / s
            return res * self.d ** (1 / alpha)
        else:
            return super().sample(N, option, **kwargs)
