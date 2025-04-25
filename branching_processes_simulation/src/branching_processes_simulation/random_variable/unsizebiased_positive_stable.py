from collections.abc import Iterable
from typing import Any, Callable, Dict
import numpy as np
from scipy.special import gamma
from scipy.integrate import quad

from branching_processes_simulation.random_variable.vau import Vau
from branching_processes_simulation.random_variable.constant import Constant
from branching_processes_simulation.random_variable.positive_stable import PositiveStable
from branching_processes_simulation.random_variable.random_variable import RandomVariable

class UnsizebiasedPositiveStable(RandomVariable):
    _interval_a = 0
    _interval_b = +np.inf

    def __new__(cls, alpha: float, d: float=1, *args, **kwargs):
        if alpha == 1:
            return Constant(d)
        return super().__new__(cls)
    
    _a : Dict[float, Vau] = {}
    
    # alpha < 1
    def __init__(self, alpha: float, d: float=1) -> None:
        assert alpha <= 1
        self.alpha = alpha
        self.d = d

        self._stable = PositiveStable(alpha, d)
        if alpha not in self._a:
            self._a[alpha] = Vau(alpha)

    def characteristic_function(self, t: np.complex64) -> np.complex64:
        return self.laplace_transform(- 1j * t)

    def laplace_transform(self, t: np.float64) -> np.float64:
        return (1 - self.mean() * quad(lambda y: np.exp(-y**self.alpha), 0, t)[0])

    def pdf(self, x: np.float64) -> np.float64:
        raise NotImplementedError()

    def cdf(self, x: np.float64) -> np.float64:
        raise NotImplementedError()

    def mean(self) -> np.float64:
        if self.alpha < 1:
            return self.alpha / gamma(1/self.alpha)
        elif self.alpha == 1:
            return 1
        else:
            return None 

    def variance(self) -> np.float64:
        if self.alpha < 1:
            return np.infty
        else:
            return 0

    def sample(self, N: int, option='cdf', **kwargs) -> np.ndarray[float]:
        alpha = self.alpha
        if option == 'cdf':
            U = self.rng.uniform(0, 1, N)
            aTheta = self._a[alpha].sample(N, **kwargs)
            aTheta = PositiveStable.a_shifted(self.alpha, aTheta)
            res = aTheta * np.power(self.rng.gamma(1/alpha, 1, N), -(1-alpha) / alpha)
            return res
        elif option == 'mcmc':
            N_burn_in = kwargs.get('N_burn_in', 1000)
            X = self._stable.sample(N + 1 + N_burn_in, **kwargs)
            U = self.rng.uniform(0, 1, N + 1 + N_burn_in)
            for i in range(N + N_burn_in):
                if U[i] > X[i] / X[i + 1]:
                    X[i + 1] = X[i]
            return X[N_burn_in + 1:]