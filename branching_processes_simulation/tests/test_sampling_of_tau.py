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
    alpha = 0.4
    tau = Tau(alpha)
    def func(x):
        if x.shape == (1,):
            return 2*(x[0]) if x[0] < 1 else 0
            # return (x[0])**0.1
        
        res = np.zeros_like(x)
        mask = x < 1
        res[mask] = 2*(x[mask])
        # res = x**0.1
        return res
    

    N=10000
    assert np.abs(tau.function_expectation(func, N, option='integrated_tail') - 0.277) < 0.05
    # assert np.abs(tau.function_expectation(func, N, option='size_biased') - 0.27) < 0.05

    # print(tau.function_expectation(func, N, option='size_biased_ber'))
    res = tau.function_expectation(func, N, option=None)

    # plt.hist(tau.sample(N)**(0.1))
    # plt.show()

    assert np.abs(res - 0.277) < 0.05