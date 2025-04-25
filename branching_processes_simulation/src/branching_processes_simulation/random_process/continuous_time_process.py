import numpy as np
from branching_processes_simulation.random_process.random_process import RandomProcess


class ContinuousTimeRandomProcess(RandomProcess):
    def _get_profile_times(self, time:float, **kwargs):
        t_per_1 = kwargs.get('t_per_1', 10)
        m = max(int(t_per_1 * time), 1)
        return np.linspace(0, time, m, True)