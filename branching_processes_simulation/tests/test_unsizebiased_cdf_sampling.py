from matplotlib import pyplot as plt
import numpy as np
import time

from scipy.integrate import quad
from scipy.special import gamma

from branching_processes_simulation.unsizebiased_positive_stable_random_variable import ARandomVariable, UnsizebiasedPositiveStableRandomVariable 


def test_unsizebiased_mcmc_sampling():
    alpha = 0.6
    S = UnsizebiasedPositiveStableRandomVariable(alpha)

    s = S.sample(500000, option='cdf')
    plt.hist(s, bins=20)
    plt.show()

    C = S.mean()

    print(np.mean(s), S.mean())
    print(np.std(s))

    x = 1
    x1 = (1 - C * quad(lambda y: np.exp(-y**alpha), 0, x)[0])
    x2 = (np.mean(np.exp(-x*s)))
    assert np.isclose(x1, x2, rtol=5e-4), f"Expected {x1} to be close to {x2}"

