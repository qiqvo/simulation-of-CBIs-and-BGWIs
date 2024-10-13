import typing
import numpy as np
from scipy.stats import poisson

from branching_processes_simulation.random_process import RandomProcess
from branching_processes_simulation.discrete_space_process.reproduction_rv import ReproductionRandomVariable


class BGW(RandomProcess):
    def __init__(self, reproduction: ReproductionRandomVariable) -> None:
        self._reproduction = reproduction

    def generating_function(self, s: np.complex64, time: int, z: int) -> np.complex64:
        g = s
        # iterations of f
        for _ in range(time):
            g = self._reproduction.generating_function(g)
        g = g ** z
        return g

    def characteristic_function(self, t: np.complex64) -> np.complex64:
        return self.generating_function(np.exp(1j * t))

    def laplace_transform(self, t: np.float64) -> np.float64:
        return np.real(self.generating_function(np.exp(-t)))
    
    def mean(self, t: np.float64, time: int, z: int) -> np.float64:
        return z

    # TODO: check:
    def variance(self, t: np.float64, time: int, z: int) -> np.float64:
        if self.alpha < 1:
            return np.infty
        else: 
            return time * self._reproduction.variance()
    
    def sample_profile(self, time: int, z: int) -> np.ndarray[int]:
        profile = np.zeros(time, int)
        profile[0] = z
        for i in range(1, time):
            profile[i] = np.sum(self._reproduction.sample(profile[i - 1]))
            if profile[i] == 0:
                break
        return profile

    def sample(self, N: int, time: int, z: int) -> np.ndarray[float]:
        # self.
        return None
    
    # TODO: implement other methods? 
    def sample_function(self, N: int, theta: typing.Callable, time: int, z: int) -> np.ndarray[float]:
        return super().sample_function(N, theta, time, z)
