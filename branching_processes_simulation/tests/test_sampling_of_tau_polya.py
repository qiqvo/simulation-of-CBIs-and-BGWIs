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
    N=10000
    alpha = 0.6

    s = FejerDeLaValleePoussinRandomVariable()
    p = Tau(alpha)
    pt = PolyaTransformedTau(alpha)

    s_sample = s.sample(N)
    # p_sample = p.sample(N)
    pt_sample = pt.sample(N)

    b = s_sample / pt_sample

    ls = np.linspace(0, 4)
    
    res = []
    for l in ls:
        res.append(np.exp(- l * np.abs(b)).mean())

    plt.plot(ls, res)
    plt.plot(ls, p.characteristic_function(ls))
    plt.show()

    assert False