from typing import Any, Callable
import numpy as np
import scipy

from branching_processes_simulation.random_variable.constant import Constant
from branching_processes_simulation.random_variable.stable import Stable


class PositiveStable(Stable):
    _interval_a = 0
    _interval_b = +np.inf
    
    def __new__(cls, alpha: float, d: float=1, *args, **kwargs):
        if alpha == 1:
            return Constant(d)
        return super().__new__(cls)
    
    # alpha < 1
    def __init__(self, alpha: float, d: float=1) -> None:
        assert alpha <= 1
        super().__init__(alpha, 1, d)

    def characteristic_function(self, t: np.complex64) -> np.complex64:
        return np.exp(- self.d * np.power(- 1j * t, self.alpha))

    def laplace_transform(self, t: np.float64) -> np.float64:
        return np.exp(- self.d * np.power(t, self.alpha))

    def pdf(self, x: np.float64) -> np.float64:
        return None # unknown

    def cdf(self, x: np.float64) -> np.float64:
        return None # unknown

    def mean(self) -> np.float64:
        if self.alpha < 1:
            return np.infty
        elif self.alpha == 1:
            return self.d
        else:
            return None 

    def variance(self) -> np.float64:
        if self.alpha < 1:
            return np.infty
        else:
            return 0

    @staticmethod
    def a_shifted(alpha, theta: np.ndarray[float]):
        res = np.sin(alpha * theta)**alpha
        res /= np.sin(theta)
        res **= 1/(1 - alpha)
        res *= np.sin((1 - alpha) * theta)
        res **= ((1 - alpha) / alpha)
        return res

    @staticmethod
    def a(alpha, beta, theta: np.ndarray[float]):
        return PositiveStable.a_shifted(alpha, theta + np.pi/2)
    
    def sample(self, N: int, option='CMS', **kwargs) -> np.ndarray[float]:
        alpha = self.alpha
        if option=='CMS': ## Kanter algo for totally skewed (beta=1)  
            theta, w = self.rng.uniform(0, 1, (2, N))
            w = -np.log(w)
            res = self.a_shifted(alpha, theta * np.pi)
            res *= w**(-(1 - alpha) / alpha) * (self.d**(1/alpha))
            # print('pos a:', PositiveStableRandomVariable.a(alpha, np.pi/2+0.1))
        elif option == 'gen_CMS' or option == 'scipy':
            if option.startswith('gen_'):
                option = option[4:]
            res = super().sample(N, option=option) # * (1 + np.tan(np.pi/2 * alpha * (1 - alpha)))**(-1/alpha)
        else:
            raise NotImplementedError(f"Option {option} is not implemented")
        return res