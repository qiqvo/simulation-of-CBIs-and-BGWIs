import numpy as np

from branching_processes_simulation.continuous_space_process.cb import CB
from branching_processes_simulation.linnik import Linnik
from branching_processes_simulation.random_process import RandomProcess


class CBI(RandomProcess):
    def __init__(self, alpha: np.float64, c: np.float64, d: np.float64) -> None:
        self.alpha = alpha
        self.c = c
        self.d = d
        self.delta = d / (alpha * c)
        self._cb = CB(alpha, c)
        self._linnik = Linnik(self.alpha, self.delta)

    def characteristic_function(self, t: np.complex64, time: np.float64, z: np.float64) -> np.complex64:
        return self.laplace_transform(-1j * t, time, z)

    def laplace_transform(self, t: np.float64, time: float, z: np.float64) -> np.float64:
        return (1 + self.alpha * self.c * np.abs(t)**self.alpha * time)**(-self.delta) * self._cb.laplace_transform(t, time, z)

    def mean(self, t: np.float64, time: float, z: np.float64) -> np.float64:
        if self.alpha < 1:
            return np.infty
        else: 
            return z + self.delta * time

    def variance(self, t: np.float64, time: float, z: np.float64) -> np.float64:
        if self.alpha < 1:
            return np.infty
        else: 
            return 2 * self.c

    def sample(self, N: int, time: float, z: np.float64) -> np.ndarray[float]:
        s = self._cb.sample(N, time, z)
        s = s + self._linnik.sample(N) * (self.alpha *self.c * time)**(1 / self.alpha)
        return s