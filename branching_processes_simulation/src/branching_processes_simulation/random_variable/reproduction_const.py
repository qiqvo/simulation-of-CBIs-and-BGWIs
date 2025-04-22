
from branching_processes_simulation.random_variable.immigration_const import ImmigrationConst
from branching_processes_simulation.random_variable.reproduction import ReproductionSL


class ReproductionConst(ReproductionSL):
    def __init__(self, alpha, c) -> None:
        assert(0 < c * (1 + alpha) < 1)
        super().__init__(alpha, c, lambda x: 1)

    def _create_immigration(self):
        immigration = ImmigrationConst(self.alpha, self.c * (1 + self.alpha))
        return immigration
