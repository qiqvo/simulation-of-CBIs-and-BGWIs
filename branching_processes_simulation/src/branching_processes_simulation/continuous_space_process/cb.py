import typing
import numpy as np
from scipy.stats import poisson

from branching_processes_simulation.random_process import RandomProcess


class CB(RandomProcess):
    def __init__(self, reproduction_mechanism: typing.Callable) -> None:
        self._reproduction_mechanism = reproduction_mechanism

    def characteristic_function(self, t: np.complex64, time: np.float64, z: np.float64) -> np.complex64:
        return self.laplace_transform(-1j * t, time, z)

    def laplace_transform(self, t: np.float64, time: float, z: np.float64) -> np.float64:
        return np.exp(-t*z / np.power(1 + self.alpha * self.c * time * np.abs(t)**self.alpha, 1 / self.alpha))

    def mean(self, time: float, z: np.float64) -> np.float64:
        return z
