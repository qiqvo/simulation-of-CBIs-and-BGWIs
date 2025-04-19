from typing import Any, Callable
import numpy as np
import scipy

from branching_processes_simulation.random_variable import RandomVariable


class StableRandomVariable(RandomVariable):
    def __init__(self, alpha: float, beta:float, d: float=1) -> None:
        assert 0 < alpha <= 2 and alpha != 1 and d > 0 and -1 <= beta <= 1

        self.alpha = alpha
        self.d = d
        self.beta = beta
        
        self._s = scipy.stats.levy_stable(alpha=alpha, beta=beta, loc=0, scale=1)
        self._s.random_state = self.rng

    def characteristic_function(self, t: np.float64) -> np.complex64:
        a1 = 1
        if self.beta != 0:
            re_t = np.real(t)
            sign_t = re_t / np.abs(re_t)

            a1 = np.tan(np.pi * self.alpha / 2) * sign_t * self.beta
            a1 = 1  - a1 * 1j
            
        return np.exp(-self.d * np.power(np.abs(t), self.alpha) * a1) 

    def laplace_transform(self, t: np.float64) -> np.float64:
        return np.real(self.characteristic_function(-t * 1j))

    def pdf(self, x: np.float64) -> np.float64:
        return None # unknown

    def cdf(self, x: np.float64) -> np.float64:
        return None # unknown

    def mean(self) -> np.float64:
        return None

    def variance(self) -> np.float64:
        return None
    
    def _k(self):
        return 1 - np.abs(1 - self.alpha)

    def sample(self, N: int, option='scipy', **kwargs) -> np.ndarray[float]:
        alpha = self.alpha
        if option == 'scipy':
            res = self._s.rvs(size=N) * np.cos(np.pi * alpha / 2)**(1/alpha)
        elif option == 'CMS':
            Phi = self.rng.uniform(-np.pi / 2, np.pi / 2, N)
            Phi_0 = -np.pi / 2 * self.beta * (self._k() / alpha)
            W = -np.log(self.rng.uniform(0, 1, N))

            res = np.sin(alpha * (Phi - Phi_0)) / (np.cos(Phi))**(1/alpha) \
                    * (np.cos(Phi - alpha * (Phi - Phi_0)) / W) ** ((1 - alpha) / alpha)
        return self.d ** 1/self.alpha * res