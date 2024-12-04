from typing import Callable
import numpy as np

from branching_processes_simulation.continuous_space_process.cb import CB
from branching_processes_simulation.random_process import RandomProcess


class CBI(RandomProcess):
    def __init__(self, reproduction_mechanism: Callable, immigration_mechanism: Callable) -> None:
        self._immigration_mechanism = immigration_mechanism
        self._reproduction_mechanism = reproduction_mechanism
        self._cb = CB(self._reproduction_mechanism)

    def characteristic_function(self, t: np.complex64, time: np.float64, z: np.float64) -> np.complex64:
        return self.laplace_transform(-1j * t, time, z)

    def laplace_transform(self, t: np.float64, time: float, z: np.float64) -> np.float64:
        return (1 + self.alpha * self.c * np.abs(t)**self.alpha * time)**(-self.delta) * self._cb.laplace_transform(t, time, z)

    def mean(self, time: float, z: np.float64) -> np.float64:
        if self.alpha < 1:
            return np.infty
        else: 
            return z + self.delta * time

    def variance(self, time: float, z: np.float64) -> np.float64:
        if self.alpha < 1:
            return np.infty
        else: 
            return 2 * self.c

    def sample(self, N: int, time: float, z: np.float64) -> np.ndarray[float]:
        s = self._cb.sample(N, time, z)
        s = s + self._linnik.sample(N) * (self.alpha *self.c * time)**(1 / self.alpha)
        return s