from abc import abstractmethod
from typing import List, Callable
import numpy as np

from branching_processes_simulation.continuous_space_process.continuous_time_process import ContinuousTimeRandomProcess


class CB(ContinuousTimeRandomProcess):
    def __init__(self, reproduction_mechanism: Callable) -> None:
        self._reproduction_mechanism = reproduction_mechanism

    def characteristic_function(self, t: np.complex64, time: np.float64, z: np.float64) -> np.complex64:
        return self.laplace_transform(-1j * t, time, z)

    def laplace_transform(self, t: np.float64, time: float, z: np.float64) -> np.float64:
        # TODO: 
        return None

    def mean(self, time: float, z: np.float64) -> np.float64:
        return z

    def sample_profile(self, time: float, z: float, t_per_1:int=10, **kwargs) -> np.ndarray[int]:
        m = max(t_per_1 * time, 2)
        times = np.linspace(0, time, m, True)

        profile = [z]
        for i in range(1, m):
            dt = times[i]
            profile.append(self.sample_on_time(1, dt, profile[-1], **kwargs)[0])

        return np.array(profile)
    
    @abstractmethod
    def sample_on_time(self, N: int, time: float, z: np.float64, **kwargs) -> np.ndarray[float]:
        return None
