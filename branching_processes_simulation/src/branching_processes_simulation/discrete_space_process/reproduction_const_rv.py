
from branching_processes_simulation.discrete_space_process.immigration_const_rv import ImmigrationConstRandomVariable
from branching_processes_simulation.discrete_space_process.reproduction_rv import ReproductionRandomVariable


class ReproductionConstRandomVariable(ReproductionRandomVariable):
    def __init__(self, alpha, c) -> None:
        assert(0 < c * (1 + alpha) < 1)
        super().__init__(alpha, c, lambda x: 1)

    def _create_immigration(self):
        immigration = ImmigrationConstRandomVariable(self.alpha, self.c * (1 + self.alpha))
        return immigration
