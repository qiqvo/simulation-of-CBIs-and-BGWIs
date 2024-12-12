from matplotlib import pyplot as plt
import numpy as np
import time

from branching_processes_simulation.stable_random_variable import PositiveStableRandomVariable




def test():
    alpha = 0.5
    N = 10000
    s = PositiveStableRandomVariable(alpha)
    sampled = s.sample(N, 'scipy')

    l = 1
    # def func(x):
    #     return np.exp(1j * l * x)

    # r = np.exp(-l**alpha * (1 - 1j * 1 * np.tan(np.pi * alpha / 2)))

    # def func(x):
    #     return np.exp(1j * l * x)

    # r = np.exp(-l**alpha * np.exp(-np.pi/2 * 1j * 1 * (1 - (1-alpha))))

    def func(x):
        return np.exp(-l * x)

    r = np.exp(-l**alpha)

    # assert np.abs(s.function_expectation(func, N, option='CMS') - r) < 0.02
    # assert np.abs(s.function_expectation(func, N, option='polya') - r) < 0.01
    # r = np.exp(-l**alpha * (1 - 1j * 1 * np.tan(np.pi * alpha / 2)))
    r = np.exp(-l**alpha)
    # r = np.exp(-l**alpha / np.cos(np.pi * alpha / 2))
    assert np.abs(func(sampled).mean() - r) < 0.001

    d = np.exp(-l * sampled).mean()
    print(-np.log(d) / l**(alpha))
    # assert False