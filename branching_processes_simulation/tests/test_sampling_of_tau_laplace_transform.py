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
    alpha = 0.6
    tau = Tau(alpha)

    def func(l):
        return lambda x: np.exp(-l * x)
        # return np.exp(-l * x) * x
    
    # def der_func(x):
    #     return -l * np.exp(-l * x)
    der_func = None

    # N=1000000
    # assert np.abs(tau.function_expectation(func, N, option='integrated_tail') - 0.6316) < 0.05
    # assert np.abs(tau.function_expectation(func, N, option='size_biased') - 0.27) < 0.05

    # print(tau.function_expectation(func, N, option='size_biased_ber'))

    ls = np.linspace(0, 10)
    res, res1, res2, res3 = [], [], [], []

    for l in ls:
        res1.append(tau.function_expectation(func(l), 10000, option='integrated_tail', theta_diff=der_func))
        res2.append(tau.function_expectation(func(l), 10000, option='polya', theta_diff=der_func))
        res3.append(tau.function_expectation(func(l), 10000, option='size_biased', theta_diff=der_func))

    plt.plot(ls, res1, label='integrated_tail')
    plt.plot(ls, res2, label='polya')
    plt.plot(ls, res3, label='size_biased')
    # plt.plot(t, res4, label='approx')
    # r = 1 - (l**alpha / (1 + l**alpha))**(1/alpha)
    # r = (1 / (1 + l**alpha))**(1 + 1/alpha)

    plt.plot(ls, 1 - (ls**alpha / (1 + ls**alpha))**(1/alpha), c='r')
    plt.legend()
    plt.show()

    # res = tau.function_expectation(func, N, option='integrated_tail', theta_diff=der_func)
    # assert np.abs(res - 1.1296) < 1e-3
    
    # res = tau.function_expectation(func, N, option=None)
    # assert np.abs(res - 1.03942) < 1e-4

    assert False
    

# if __name__ == '__main__':
#     test()