
from branching_processes_simulation.random_variable.immigration_exp import ImmigrationExp
from branching_processes_simulation.random_variable.reproduction import ReproductionSL


class ReproductionExp(ReproductionSL):
    def __init__(self, alpha, c) -> None:
        super().__init__(alpha, c, ImmigrationExp.create_k(alpha, c * (1 + alpha)))

    def _create_immigration(self):
        immigration = ImmigrationExp(self.alpha, self.c * (1 + self.alpha))
        return immigration
