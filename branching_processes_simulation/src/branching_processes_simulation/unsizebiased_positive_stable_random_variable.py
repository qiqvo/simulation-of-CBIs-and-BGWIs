from collections.abc import Iterable
from typing import Any, Callable, Dict
import numpy as np
from scipy.special import gamma, hyp1f1, gammaincc
from scipy.integrate import quad

from branching_processes_simulation.constant_variable import ConstantVariable
from branching_processes_simulation.positive_stable_random_variable import PositiveStableRandomVariable
from branching_processes_simulation.random_variable import RandomVariable
from branching_processes_simulation.utils import parallel_integrate_upper_limits

class ARandomVariable(RandomVariable):
    _interval_a = 0
    _interval_b = np.pi

    def __init__(self, alpha: float) -> None:
        super().__init__()
        self.alpha = alpha
    
    def pdf(self, x: np.float64) -> np.float64:
        res = self.alpha / np.pi / PositiveStableRandomVariable.a(self.alpha, 1, x)
        return res

    def cdf(self, x: np.float64) -> np.float64:
        if isinstance(x, Iterable):
            return parallel_integrate_upper_limits(self.pdf, 0, x)
        else:
            return quad(self.pdf, 0, x)[0]
    
    def sample(self, N, **kwargs):
        return self.sample_from_cdf(N, True, approximation='linear', **kwargs)
    
    def characteristic_function(self, t):
        return super().characteristic_function(t)
    
    def laplace_transform(self, t):
        return super().laplace_transform(t)
    
    def mean(self) -> np.float64:
        return super().mean()
    
    def variance(self) -> np.float64:  
        return super().variance()

class UnsizebiasedPositiveStableRandomVariable(RandomVariable):
    def __new__(cls, alpha: float, d: float=1, *args, **kwargs):
        if alpha == 1:
            return ConstantVariable(d)
        return super().__new__(cls)
    
    _a : Dict[float, ARandomVariable] = {}
    
    # alpha < 1
    def __init__(self, alpha: float, d: float=1) -> None:
        assert alpha <= 1
        self.alpha = alpha
        self.d = d

        self._stable = PositiveStableRandomVariable(alpha, d)
        if alpha not in self._a:
            self._a[alpha] = ARandomVariable(alpha)

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
            A = self._a[alpha].sample(N, **kwargs)
            aTheta = PositiveStableRandomVariable.a(self.alpha, 1, A)
            res = aTheta * np.power((1 / self.rng.gamma(1/alpha, 1, N)), (1-alpha) / alpha)
            return res
        elif option == 'mcmc':
            N_burn_in = kwargs.get('N_burn_in', 1000)
            X = self._stable.sample(N + 1 + N_burn_in, **kwargs)
            U = self.rng.uniform(0, 1, N + 1 + N_burn_in)
            for i in range(N + N_burn_in):
                if U[i] > X[i] / X[i + 1]:
                    X[i + 1] = X[i]
            return X[N_burn_in + 1:]