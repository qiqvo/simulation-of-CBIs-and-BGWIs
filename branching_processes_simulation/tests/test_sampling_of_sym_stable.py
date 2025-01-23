from matplotlib import pyplot as plt
import numpy as np
import time

from branching_processes_simulation.positive_stable_random_variable import PositiveStableRandomVariable
from branching_processes_simulation.symmetric_stable_random_variable import SymmetricStableRandomVariable




def test_polya():
    alpha = 0.77
    N = int(1E6)
    s = SymmetricStableRandomVariable(alpha)
    sampled = s.sample(N, 'polya')

    ls = [0.1, 0.3, 0.7, 1, 2, 10]
    data = [np.cos(l * sampled).mean() for l in ls]
    r = [np.exp(-l**alpha) for l in ls]

    # plt.scatter(ls, data)
    # plt.plot(ls, r)
    # plt.show()

    # print('Estimated c:', np.mean([-np.log(data)/l**alpha for l in ls]))

    assert np.allclose(data, r, rtol=0.05)
    

def test_scipy():
    alpha = 0.77
    N = int(1E5)
    s = SymmetricStableRandomVariable(alpha)
    sampled = s.sample(N, 'scipy')

    ls = [0.1, 0.3, 0.7, 1, 2, 10]
    data = [np.cos(l * sampled).mean() for l in ls]
    r = [np.exp(-l**alpha) for l in ls]

    # plt.scatter(ls, data)
    # plt.plot(ls, r)
    # plt.show()

    # print('Estimated c:', np.mean([-np.log(data)/l**alpha for l in ls]))

    assert np.allclose(data, r, rtol=0.05)
    
