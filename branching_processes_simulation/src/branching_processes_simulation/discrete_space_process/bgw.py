import typing
import numpy as np

from branching_processes_simulation.discrete_space_process.discrete_time_process import DiscreteTimeRandomProcess
from branching_processes_simulation.discrete_space_process.reproduction_rv import ReproductionRandomVariable
from branching_processes_simulation.discrete_space_process.genealogy.node import Node


class BGW(DiscreteTimeRandomProcess):
    def __init__(self, reproduction: ReproductionRandomVariable) -> None:
        self._reproduction = reproduction

    def get_reproduction_sample(self, N: int):
        return self._reproduction.sample(N)

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
    
    def mean(self, time: int, z: int) -> np.float64:
        return self._reproduction.mean()**time * z

    def variance(self, time: int, z: int) -> np.float64:
        m = self._reproduction.mean()
        if m == 1:
            res = time
        else:
            res = m**(time -1) * (m**time - 1) / (m - 1)
        return res * self._reproduction.variance() * z
    
    def sample_profile(self, time: int, z: int) -> np.ndarray[int]:
        profile = np.zeros(time, int)
        profile[0] = z
        for i in range(1, time):
            profile[i] = np.sum(self.get_reproduction_sample(profile[i - 1]))
            if profile[i] == 0:
                break
        return profile
    
    def sample_profile_from_genealogy(self, time: int, root: Node) -> np.ndarray[int]:
        profile = np.zeros(time, int)
        profile[0] = len(root.children)
        for e in root.children:
            profile[1] += self.count_layer(1, time, e, profile)
        return profile
        
    def count_layer(self, i: int, time: int, e: Node, profile: np.ndarray[int]) -> int:
        for child in e.children:
            profile[i + 1] += self.count_layer(i+1, time, child, profile)
        return len(e.children)