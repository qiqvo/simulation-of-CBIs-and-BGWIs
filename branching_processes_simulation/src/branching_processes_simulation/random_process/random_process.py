from abc import ABC, abstractmethod
from typing import Callable, List
import numpy as np

from branching_processes_simulation.i_random import IRandom


class RandomProcess(IRandom):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def characteristic_function(self, t: np.complex64, time: float, z: np.float64) -> np.complex64:
        return None

    @abstractmethod
    def laplace_transform(self, t: np.float64, time: float, z: np.float64) -> np.float64:
        return None

    @abstractmethod
    def mean(self, time: float, z: np.float64) -> np.float64:
        return None

    @abstractmethod
    def variance(self, time: float, z: np.float64) -> np.float64:
        return None

    ## Returns a sample in a shape (len(z), N) 
    @abstractmethod
    def sample(self, N: int, time: np.float64, z: List[np.float64], **kwargs) -> np.ndarray[np.ndarray[np.float64]]:
        return None

    @abstractmethod
    def _get_profile_times(self, time, **kwargs):
        return None

    ## Returns a sample in a shape (N, len(times)), len(times) ~ t_per_1 * time
    def sample_profile(self, N: int, time: float, z: float, **kwargs) -> np.ndarray[int]:
        times = self._get_profile_times(time, **kwargs)
        m = len(times)

        profile = np.zeros((N, m), np.float64)
        profile[:, 0] = [z] * N
        for i in range(1, m):
            dt = times[i]
            profile[:, i] = self.sample(1, dt, profile[:, i - 1], **kwargs)[:, 0]

        return profile