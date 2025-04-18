from matplotlib import pyplot as plt
import numpy as np
import time

from branching_processes_simulation.continuous_space_process.fejer_de_la_vallee_poussin_random_variable import FejerDeLaValleePoussinRandomVariable
from branching_processes_simulation.continuous_space_process.tau import Tau
from branching_processes_simulation.discrete_space_process.bgw import BGW
from branching_processes_simulation.discrete_space_process.bgwi import BGWI
from branching_processes_simulation.discrete_space_process.immigration_exp_rv import ImmigrationExpRandomVariable
from branching_processes_simulation.discrete_space_process.reproduction_exp_rv import ReproductionExpRandomVariable




def test():
    N=100000
    s = FejerDeLaValleePoussinRandomVariable()

    sampled = s.sample(N)
    assert np.abs((np.abs(sampled)**0.4).mean() - 1.38) < 0.01


    ls = np.linspace(0, 4)
    
    res = []
    for l in ls:
        res.append(np.exp(-l* np.abs(sampled)).mean())

    plt.plot(ls, res)
    plt.show()

    assert False