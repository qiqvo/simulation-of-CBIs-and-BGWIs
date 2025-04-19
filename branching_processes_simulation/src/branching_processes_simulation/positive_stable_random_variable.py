from typing import Any, Callable
import numpy as np
import scipy

from branching_processes_simulation.constant_variable import ConstantVariable
from branching_processes_simulation.stable_random_variable import StableRandomVariable


class PositiveStableRandomVariable(StableRandomVariable):
    def __new__(cls, alpha: float, d: float=1, *args, **kwargs):
        if alpha == 1:
            return ConstantVariable(d)
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

    def _a(self, theta: np.ndarray[float]):
        if self.alpha > 0.99:
            sin_th = np.sin(theta)
            cot_th = np.sqrt(1 - sin_th**2)/sin_th
            a = 1 - self.alpha

            res = a*theta \
                - theta*(theta*cot_th + np.log(sin_th))*a**2 \
                + (-(2*theta**3)/3 + theta*(2*theta*cot_th \
                + np.log(sin_th) - 2)*np.log(np.sqrt(sin_th)))*a**3
        else:
            res = np.sin(self.alpha * theta)**self.alpha
            res /= np.sin(theta)
            res **= 1/self.alpha
            res = np.sin((1 - self.alpha) * theta) * res
        return res

    def sample(self, N: int, option='scipy', **kwargs) -> np.ndarray[float]:
        alpha = self.alpha
        if option=='CMS': ## Kanter algo for totally skewed (beta=1)  
            theta, w = self.rng.uniform(0, 1, (2, N))
            w = -np.log(self.rng.uniform(0, 1, N))
            a = self._a(theta * np.pi)
            res = np.power(a / w, (1 - alpha) / alpha) * ((self.d)**(1/alpha))
        elif option == 'gen_CMS' or option == 'scipy':
            if option.startswith('gen_'):
                option = option[4:]
            res = super().sample(N, option=option) * (np.cos(np.pi * alpha /2))**(1/alpha)

        return res