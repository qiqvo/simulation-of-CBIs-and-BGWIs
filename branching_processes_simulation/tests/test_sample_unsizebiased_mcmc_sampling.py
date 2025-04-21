from matplotlib import pyplot as plt
import numpy as np
import time

from scipy.integrate import quad
from scipy.special import gamma

from branching_processes_simulation.unsizebiased_positive_stable_random_variable import UnsizebiasedPositiveStableRandomVariable 

def test_unsizebiased_mcmc_sampling():
    alpha = 0.6
    S = UnsizebiasedPositiveStableRandomVariable(alpha)

    s = S.sample(500000, option='mcmc', N_burn_in=30000)
    # plt.hist(s, bins=100)
    # plt.show()

    C = S.mean()

    print(np.mean(s), S.mean())
    print(np.std(s))

    x = 1
    x1 = S.laplace_transform(x)
    x2 = (np.mean(np.exp(-x*s)))
    assert np.isclose(x1, x2, rtol=0.05), f"Expected {x1} to be close to {x2}"
