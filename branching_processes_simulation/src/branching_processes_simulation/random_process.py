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

    @abstractmethod
    def sample(self, N: int, times: List[float], z: np.float64, function:Callable=None, **kwargs) -> np.ndarray[np.ndarray[float]]:
        if len(times) == 1:
            res = np.array([self.sample_on_time(N, times[0], z, **kwargs)])
        else:
            res = self.sample_on_times(N, times, z, **kwargs)
        
        if function is not None:
            for i in range(len(times)):
                res[i] = function(res[i], t=times[i])
        return res

    def sample_on_time(self, N: int, time: float, z: np.float64, **kwargs) -> np.ndarray[float]:
        return self.sample_on_times(N, [time], z, **kwargs).flatten()
    
    @abstractmethod
    def sample_on_times(self, N: int, times: List[float], z: np.float64, **kwargs) -> np.ndarray[np.ndarray[float]]:
        return None
