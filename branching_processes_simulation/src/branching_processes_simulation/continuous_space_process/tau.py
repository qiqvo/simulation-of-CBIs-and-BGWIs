from typing import Callable
import numpy as np
from sympy import gamma

from branching_processes_simulation.exponential import Exponential
from branching_processes_simulation.linnik import Linnik
from branching_processes_simulation.positive_stable_random_variable import PositiveStableRandomVariable
from branching_processes_simulation.random_variable import RandomVariable


class Tau(RandomVariable):
    def __new__(cls, alpha: float, *args, **kwargs):
        if alpha == 1:
            return Exponential(1)
        return super().__new__(cls)

    def __init__(self, alpha: float) -> None:
        assert 0 < alpha <= 1

        self.alpha = alpha
        self._stable = PositiveStableRandomVariable(alpha)
        self._linnik_0 = Linnik(alpha, 1/alpha)
        self._linnik_1 = Linnik(alpha, 1 + 1/alpha)
        self._ber = lambda p, size: self.rng.binomial(n=1, p=p, size=size)

    def characteristic_function(self, t: np.complex64) -> np.complex64:
        return 1 - np.power(np.power(np.abs(t), self.alpha) / (1 + np.power(np.abs(t), self.alpha)),1/self.alpha)

    def laplace_transform(self, t: np.float64) -> np.float64:
        return np.real(self.characteristic_function(t))

    def pdf(self, x: np.float64) -> np.float64:
        return None

    def cdf(self, x: np.float64) -> np.float64:
        return None

    def mean(self) -> np.float64:
        return 1

    def variance(self) -> np.float64:
        return np.infty

    def sample(self, N: int, option='cdf', **kwargs) -> np.ndarray[float]:
        if option == 'cdf':
            alpha = self.alpha
            X = self._stable.sample(N)
            U = self.rng.uniform(0, 1, N)
            Y = X * (-np.log(-(U * X * gamma(1/alpha))/alpha))^(1/alpha)
            return Y            
        
        raise NotImplementedError()
    
    def function_expectation(self, theta: Callable, N=100, option='integrated_tail', theta_diff=None, **kwargs) -> np.ndarray[float]:
        if option == 'integrated_tail':
            delta = 0.001
            if theta_diff is None:
                theta_diff = lambda x: (theta(x + delta) - theta(x)) / delta
            res = theta(0) + self._linnik_0.sample_function(N, theta_diff, **kwargs).mean()
        elif option == 'size_biased_ber':
            s = self._linnik_1.sample(N)
            for i in range(N):
                h = self._ber(1/(s[i] + 1), 1)
                while h == 0:
                    s[i] = self._linnik_1.sample(1)[0]
                    h = self._ber(1/(s[i] + 1), 1)
                s[i] = theta(np.array([s[i]])) * (s[i] + 1) / s[i]
            res = s.mean()
        elif option == 'size_biased':
            s = self._linnik_1.sample(N)
            s = theta(s) / s
            res = s.mean()
        return res