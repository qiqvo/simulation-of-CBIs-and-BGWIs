import typing
import numpy as np

from branching_processes_simulation.discrete_space_process.genealogy.node import Node
from branching_processes_simulation.discrete_space_process.reproduction_rv import ReproductionRandomVariable
from branching_processes_simulation.discrete_space_process.immigration_rv import ImmigrationRandomVariable
from branching_processes_simulation.discrete_space_process.bgw import BGW


class BGWI(BGW):
    def __init__(self, reproduction: ReproductionRandomVariable, immigration: ImmigrationRandomVariable) -> None:
        super().__init__(reproduction)
        self._immigration = immigration

    def get_immigration_sample(self, N: int):
        return self._immigration.sample(N)

    def generating_function(self, s: np.complex64, time: int, z: int) -> np.complex64:
        return None

    def characteristic_function(self, t: np.complex64) -> np.complex64:
        return self.generating_function(np.exp(1j * t))

    def laplace_transform(self, t: np.float64) -> np.float64:
        return np.real(self.generating_function(np.exp(-t)))
    
    def mean(self, time: int, z: int) -> np.float64:
        m = self._reproduction.mean()
        if m == 1:
            res = time 
        else:
            res = m * (m**time - 1) / (m - 1)

        return res * self._immigration.mean() + m**time * z

    def variance(self, time: int, z: int) -> np.float64:
        m = self._reproduction.mean()
        v = self._reproduction.variance()
        mj = self._immigration.mean()
        vj = self._immigration.variance()

        if m == 1:
            res = time * vj + v*m* time * (time - 1) / 2 + time*v*z
        else:
            t = (m**(2*time) - 1) / (m**2 - 1)
            res = vj*t + v*mj*(m**(time-1) (m**time - 1)/(m-1) - t) / (m-1)
            res += m**(time -1) * (m**time - 1) / (m - 1) *v*z
        
        return res
    
    def sample_profile(self, time: int, z: int) -> np.ndarray[int]:
        profile = np.zeros((time, time), int)
        immigrants = self.get_immigration_sample(time)
        profile[0, :] = super().sample_profile(time, z + immigrants[0])
        for i in range(1, time):
            profile[i, i:] = super().sample_profile(time-i, immigrants[i])
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