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
    alpha = 0.9
    tau = Tau(alpha)
    l = 0.01
    def func(x):
        return np.exp(-l * x) # * x
        # return np.exp(-l * x) * x
    
    # def der_func(x):
    #     return -l * np.exp(-l * x)
    der_func = None

    # N=1000000
    # assert np.abs(tau.function_expectation(func, N, option='integrated_tail') - 0.6316) < 0.05
    # assert np.abs(tau.function_expectation(func, N, option='size_biased') - 0.27) < 0.05

    # print(tau.function_expectation(func, N, option='size_biased_ber'))

    t, res1, res2, res3 = [], [], [], []
    for i in range(10, 50):
        t.append(i*500)
        res1.append(tau.function_expectation(func, i*500, option='integrated_tail', theta_diff=der_func))
        res2.append(tau.function_expectation(func, i*500, option=None, theta_diff=der_func))
        res3.append(tau.function_expectation(func, i*500, option='size_biased', theta_diff=der_func))

    # res4 = [(res1[i] + res2[i] + res3[i])/3 for i in range(len(t))]

    plt.plot(t, res1, label='int tail')
    plt.plot(t, res2, label='Polya')
    # plt.plot(t, res3, label='size_biased')
    # plt.plot(t, res4, label='approx')
    r = 1 - (l**alpha / (1 + l**alpha))**(1/alpha)
    # r = (1 / (1 + l**alpha))**(1 + 1/alpha)

    plt.plot([t[0], t[-1]], [r, r], c='r')
    plt.legend()
    plt.show()

    # res = tau.function_expectation(func, N, option='integrated_tail', theta_diff=der_func)
    # assert np.abs(res - 1.1296) < 1e-3
    
    # res = tau.function_expectation(func, N, option=None)
    # assert np.abs(res - 1.03942) < 1e-4

    assert False
    

# if __name__ == '__main__':
#     test()