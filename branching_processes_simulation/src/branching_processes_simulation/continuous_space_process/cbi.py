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
        ## TODO: 
        return None

    def mean(self, time: float, z: np.float64) -> np.float64:
        ## TODO: 
        return None

    def variance(self, time: float, z: np.float64) -> np.float64:
        ## TODO: 
        return None