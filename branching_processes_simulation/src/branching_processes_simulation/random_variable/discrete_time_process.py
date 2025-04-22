from branching_processes_simulation.random_process.random_process import RandomProcess


class DiscreteTimeRandomProcess(RandomProcess):
    def _get_profile_times(self, time:int, **kwargs):
        return list(range(time))