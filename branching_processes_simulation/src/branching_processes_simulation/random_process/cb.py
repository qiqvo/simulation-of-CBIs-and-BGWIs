from abc import abstractmethod
from typing import List, Callable
import numpy as np

from branching_processes_simulation.random_process.continuous_time_process import (
    ContinuousTimeRandomProcess,
)


class CriticalCB(ContinuousTimeRandomProcess):
    def __init__(self, reproduction_mechanism: Callable) -> None:
        self._reproduction_mechanism = reproduction_mechanism

    def characteristic_function(
        self, t: np.complex64, time: np.float64, z: np.float64
    ) -> np.complex64:
        raise NotImplementedError()
        # return self.laplace_transform(-1j * t, time, z)

    def laplace_transform(
        self, t: np.float64, time: float, z: np.float64
    ) -> np.float64:
        raise NotImplementedError()

    def mean(self, time: float, z: np.float64) -> np.float64:
        return z

    def variance(self, time: float, z: np.float64) -> np.float64:
        raise NotImplementedError()

    def sample(
        self, N: int, time: np.float64, z: List[np.float64], **kwargs
    ) -> np.ndarray[np.ndarray[float]]:
        raise NotImplementedError()
