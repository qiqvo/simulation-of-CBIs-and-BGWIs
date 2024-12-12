from typing import Any, Callable
import numpy as np
import scipy

from branching_processes_simulation.constant_variable import ConstantVariable
from branching_processes_simulation.continuous_space_process.fejer_de_la_vallee_poussin_random_variable import FejerDeLaValleePoussinRandomVariable
from branching_processes_simulation.random_variable import RandomVariable


class PositiveStableRandomVariable(RandomVariable):
    def __new__(cls, alpha: float, d: float=1, *args, **kwargs):
        if alpha == 1:
            return ConstantVariable(d)
        return super().__new__(cls)
    
    # alpha < 1
    def __init__(self, alpha: float, d: float=1) -> None:
        assert 0 < alpha <= 1 and d > 0
        self.alpha = alpha
        self.d = d
        self._fvp = FejerDeLaValleePoussinRandomVariable()
        self._s = scipy.stats.levy_stable(alpha=alpha, beta=1, loc=0, scale=d)
        self._s.random_state = self.rng

    def characteristic_function(self, t: np.complex64) -> np.complex64:
        return np.exp(- self.d * np.power(t, self.alpha))

    def laplace_transform(self, t: np.float64) -> np.float64:
        return np.real(self.characteristic_function(t))

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
            c2 = np.sin(self.alpha * theta)**self.alpha
            c2 /= np.sin(theta)
            c2 **= 1/self.alpha
            res = np.sin((1 - self.alpha) * theta) * c2
        return res

    def sample(self, N: int, option='scipy', **kwargs) -> np.ndarray[float]:
        alpha = self.alpha
        if option=='CMS': ## Kanter algo for totally skewed (beta=1)  
            theta = self.rng.uniform(0, np.pi, N)
            w = -np.log(self.rng.uniform(0, 1, N))
            a = self._a(theta)
            return np.power(a / w, (1 - alpha) / alpha) * ((self.d)**(1/alpha))
        elif option == 'polya':
            v = np.abs(self._fvp.sample(N))
            u1, u2 = self.rng.uniform(0, 1, (2, N))
            s = np.log(np.maximum(self.alpha / (u1 * u2), 1/u1)) ** (1 / alpha)
            return v / s * (self.d**(1/alpha))
        elif option == 'scipy':
            return self._s.rvs(size=N) * ((np.cos(np.pi * alpha /2) * self.d)**(1/alpha))
        # elif option == 'zolotarev': # 1986
        #     u1, u2 = self.rng.uniform(0, 1, (2, N))
        #     w1, w2 = np.pi * (u1 - 0.5), -np.log(u2)

