from matplotlib import pyplot as plt
import numpy as np
import time

from branching_processes_simulation.continuous_space_process.fejer_de_la_vallee_poussin_random_variable import FejerDeLaValleePoussinRandomVariable
from branching_processes_simulation.continuous_space_process.polya_transformed_tau import PolyaTransformedTau
from branching_processes_simulation.continuous_space_process.tau import Tau
from branching_processes_simulation.discrete_space_process.bgw import BGW
from branching_processes_simulation.discrete_space_process.bgwi import BGWI
from branching_processes_simulation.discrete_space_process.immigration_exp_rv import ImmigrationExpRandomVariable
from branching_processes_simulation.discrete_space_process.reproduction_exp_rv import ReproductionExpRandomVariable


def test():
    alpha=0.4
    N=10000
    v = PolyaTransformedTau(alpha)
    s = v.sample(N)
    # r = v.fractional_moment(0.1)

    assert np.abs((np.abs(s)**0.1).mean() - 1.2074) < 0.01