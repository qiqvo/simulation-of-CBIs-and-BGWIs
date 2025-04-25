from typing import Callable, List
import numpy as np
from scipy.stats import poisson

from branching_processes_simulation.random_process.stable_cbi import StableCBI
from branching_processes_simulation.random_variable.tau import Tau
from branching_processes_simulation.random_variable.zero_truncated_poisson import ZeroTruncatedPoisson

## CB conditioned to live (forever) is a CBI with d = c * (1 + alpha)
class ConditionedStableCB(StableCBI):
    def __init__(self, alpha: np.float64, c: np.float64) -> None:
        assert 0 < alpha <= 1 and c > 0
        super().__init__(alpha, c, c * (1 + alpha))
        self._xi = Tau(alpha)
        
    # def sample(self, N: int, time: np.float64, z: List[np.float64], option='poisson', **kwargs) -> np.ndarray[np.ndarray[float]]:
    #     if option == 'poisson':
    #         k = (self.alpha * self.c * time)**(1 / self.alpha)
    #         m = len(z)
    #         S = np.zeros((m, N), np.float64)
    #         for i in range(len(z)):
    #             if z[i] == 0:
    #                 continue

    #             s = ZeroTruncatedPoisson(z[i] / k).sample(N)
    #             X = self._xi.sample(np.sum(s), **kwargs) * k
    #             b = np.cumulative_sum(s[:-1], include_initial=True)
    #             S[i, :] = np.add.reduceat(X, b)
    #         return S
    #     if option=='CBI':
    #         return super().sample(N, time, z, **kwargs)
        
    #     raise ValueError(f"Unknown option: {option}")