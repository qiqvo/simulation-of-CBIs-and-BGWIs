import typing
import numpy as np
from scipy.stats import poisson

from branching_processes_simulation.discrete_space_process.genealogy.node import Node
from branching_processes_simulation.random_process import RandomProcess
from branching_processes_simulation.discrete_space_process.reproduction_rv import ReproductionRandomVariable
from branching_processes_simulation.discrete_space_process.immigration_rv import ImmigrationRandomVariable
from branching_processes_simulation.discrete_space_process.bgw import BGW


class BGWI(RandomProcess):
    def __init__(self, reproduction: ReproductionRandomVariable, immigration: ImmigrationRandomVariable) -> None:
        self._reproduction = reproduction
        self._immigration = immigration
        self._bgw = BGW(reproduction)

    def generating_function(self, s: np.complex64, time: int, z: int) -> np.complex64:
        return None

    def characteristic_function(self, t: np.complex64) -> np.complex64:
        return self.generating_function(np.exp(1j * t))

    def laplace_transform(self, t: np.float64) -> np.float64:
        return np.real(self.generating_function(np.exp(-t)))
    
    def mean(self, t: np.float64, time: int, z: int) -> np.float64:
        return z

    # TODO: check:
    def variance(self, t: np.float64, time: int, z: int) -> np.float64:
        if self.alpha < 1:
            return np.infty
        else: 
            return time * self._reproduction.variance()
    
    def sample_profile(self, time: int, z: int) -> np.ndarray[int]:
        profile = np.zeros((time, time), int)
        immigrants = self._immigration.sample(time)
        profile[0, :] = self._bgw.sample_profile(time, z + immigrants[0])
        for i in range(1, time):
            profile[i, i:] = self._bgw.sample_profile(time-i, immigrants[i])
        return profile

    def sample_profile_from_genealogy(self, time: int, root: Node) -> np.ndarray[int]:
        profile = np.zeros(time, int)
        profile[0] = len(root.children)
        for e in root.children:
            profile[1] += self.count_layer(1, time, e, profile)
        profile = profile - 1
        return profile
        
    def count_layer(self, i: int, time: int, e: Node, profile: np.ndarray[int]) -> int:
        for child in e.children:
            profile[i + 1] += self.count_layer(i+1, time, child, profile)
        return len(e.children)

    def sample_genealogy(self, time: int, z: int) -> Node:
        root = Node()
        immigrant = self._immigration.sample(1)[0]
        root.create_offspring(z + immigrant)
        for e in root.children:
            self._bgw.create_layer(time-1, e)
        # Infinite stem particle:
        root.create_offspring(1)
        self.create_layer(time - 1, root.children[-1])
        return root
    
    def create_layer(self, time: int, e: Node):
        immigrant = self._immigration.sample(1)[0]
        e.create_offspring(immigrant)
        for new_e in e.children:
            self._bgw.create_layer(time-1, new_e)
            # self.create_layer(time - 1, new_e)
        e.create_offspring(1)
        self.create_layer(time - 1, e.children[-1])

    def sample(self, N: int, time: int, z: int) -> np.ndarray[float]:
        return None
    
    # TODO: implement other methods? 
    def sample_function(self, N: int, theta: typing.Callable, time: int, z: int) -> np.ndarray[float]:
        return super().sample_function(N, theta, time, z)
