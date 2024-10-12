
from branching_processes_simulation.immigration_exp_rv import ImmigrationExpRandomVariable
from branching_processes_simulation.reproduction_rv import ReproductionRandomVariable


class ReproductionExpRandomVariable(ReproductionRandomVariable):
    def __init__(self, alpha, c) -> None:
        super().__init__(alpha, c, ImmigrationExpRandomVariable.create_k(alpha, c * (1 + alpha)))

    def _create_immigration(self):
        immigration = ImmigrationExpRandomVariable(self.alpha, self.c * (1 + self.alpha))
        return immigration
