from typing import Callable
import numpy as np

from random_variable import RandomVariable


class ImmigrationRandomVariable(RandomVariable):
    def __init__(self, alpha, d, k) -> None:
        super().__init__()
        self.alpha = alpha
        self.d = d
        self.k = k

    def generating_function(self, s: np.complex64) -> np.complex64:
        return 1 - self.d * (1 - s)**(self.alpha) * self.k(1 - s)

    def characteristic_function(self, t: np.complex64) -> np.complex64:
        return self.generating_function(np.exp(1j * t))

    def laplace_transform(self, t: np.float64) -> np.float64:
        return np.real(self.generating_function(np.exp(-t)))

    def pdf(self, x: np.float64) -> np.float64:
        return None

    def cdf(self, x: np.float64) -> np.float64:
        return None
    
    def inverse_cdf(self, x: np.float64) -> np.float64:
        return None

    def mean(self) -> np.float64:
        if self.alpha < 1:
            return np.infty
        else:
            return self.d * self.k(0)

    def variance(self) -> np.float64:
        return 

    def sample(self, N: int) -> np.ndarray[float]:
        return None

    def sample_function(self, N: int, theta: Callable) -> np.ndarray[float]:
        return None