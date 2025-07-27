import numpy as np
from branching_processes_simulation.random_process.random_process import RandomProcess


class DiscreteTimeRandomProcess(RandomProcess):
    def _get_profile_times(self, time: int, **kwargs):
        return np.arange(0, time + 1, 1)
