import numpy as np
from scipy.stats import poisson

from immigration_rv import ImmigrationRandomVariable
from stable_random_variable import StableRandomVariable


class ImmigrationExpRandomVariable(ImmigrationRandomVariable):
    def __init__(self, alpha, d) -> None:
        super().__init__(alpha, d, ImmigrationExpRandomVariable.create_k(alpha, d))
        self._s = StableRandomVariable(self.alpha)

    @staticmethod
    def create_k(alpha, d):
        k = lambda x: (np.exp(-d * x**alpha) - 1) / (d * x**alpha) if x > 0 else 1
        return k

    def sample(self, N: int) -> np.ndarray[float]:
        s = self._s.sample(N)
        s = poisson.rvs(s)
        return s