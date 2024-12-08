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
    
    def sample_on_times(self, N: int, times: List[int], z: np.float64, **kwargs) -> np.ndarray[np.ndarray[float]]:
        return np.array([self.sample_profile(times[-1], z, **kwargs)[times] for _ in range(N)]).T

    @abstractmethod
    def _get_profile_times(self, time, **kwargs):
        return None

    def sample_profile(self, time: float, z: float, **kwargs) -> np.ndarray[int]:
        times = self._get_profile_times(time, **kwargs)
        m = len(times)

        profile = [z]
        for i in range(1, m):
            dt = times[i]
            profile.append(self.sample_on_time(1, [dt], profile[-1], **kwargs)[0])

        return np.array(profile)