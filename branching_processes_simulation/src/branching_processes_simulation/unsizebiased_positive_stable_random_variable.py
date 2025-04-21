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
        res = PositiveStableRandomVariable.a(self.alpha, x)**(1 - 1/self.alpha)
        res *= self.alpha / np.pi
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
        return None # unknown

    def cdf(self, x: np.float64) -> np.float64:
        return None # unknown

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

    @staticmethod
    def f(alpha, x):
        # First term
        term1 = - ((alpha + 1) * x**((-alpha - 1) / alpha) + alpha * x**((-2*alpha - 1) / alpha))
        term1 *= alpha**4 * np.exp(-1/x) * hyp1f1(2, (4*alpha + 1) / alpha, 1/x)
        
        # Components for the second term
        A = ((x + 1/2) * alpha**2 + (3 * x * alpha) / 2 + x / 2) * alpha
        B = ((x - 1/2) * alpha + x / 2) * (alpha + 1) * ((x + 1) * alpha + x)
        
        term2 = -12 * (alpha + 1/3) * (alpha + 1/2)
        term2 *= (-x * A * gamma((2*alpha + 1) / alpha) * gammaincc((2*alpha + 1) / alpha, 1/x) + gamma((alpha + 1) / alpha) * B)
        
        # Denominator
        denom = 6 * alpha**4 + 11 * alpha**3 + 6 * alpha**2 + alpha
        
        return (term1 + term2) / denom

    def sample(self, N: int, option='mcmc', **kwargs) -> np.ndarray[float]:
        alpha = self.alpha
        if option == 'cdf':
            U = self.rng.uniform(0, 1, N)
            A = self._a[alpha].sample(N, **kwargs)
            print(A)

            print(self._a[alpha]._table)

            # x.f(alpha, x**(alpha / (1-alpha))*A)
            # res = np.power(a / w, (1 - alpha) / alpha) * ((self.d)**(1/alpha))
            aTheta = PositiveStableRandomVariable.a(self.alpha, A)
            res = np.power((aTheta / self.rng.gamma(1/alpha, 1, N)), (1-alpha) / alpha)
            return res
        elif option == 'mcmc':
            N_burn_in = kwargs.get('N_burn_in', 1000)
            X = self._stable.sample(N + 1 + N_burn_in, **kwargs)
            U = self.rng.uniform(0, 1, N + 1 + N_burn_in)
            for i in range(N + N_burn_in):
                if U[i] > X[i] / X[i + 1]:
                    X[i + 1] = X[i]
            return X[N_burn_in + 1:]