import numpy as np

from branching_processes_simulation.random_variable.immigration_sl import ImmigrationSL
from branching_processes_simulation.random_variable.positive_stable import PositiveStable


class ImmigrationExp(ImmigrationSL):
    def __init__(self, alpha, d) -> None:
        super().__init__(alpha, d, ImmigrationExp.create_k(alpha, d))
        self._s = PositiveStable(self.alpha)

    @staticmethod
    def create_k(alpha, d):
        k = lambda x: (np.exp(-d * x**alpha) - 1) / (d * x**alpha) if x > 0 else 1
        return k

    def sample(self, N: int, **kwargs) -> np.ndarray[float]:
        s = self._s.sample(N, **kwargs)
        s = self.rng.poisson(s)
        return s