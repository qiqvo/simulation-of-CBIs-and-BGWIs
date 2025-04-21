from matplotlib import pyplot as plt
import numpy as np
import time

from scipy.integrate import quad
from scipy.special import gamma

from branching_processes_simulation.unsizebiased_positive_stable_random_variable import ARandomVariable, UnsizebiasedPositiveStableRandomVariable 


def test_unsizebiased_mcmc_sampling():
    alpha = 0.2

    A = ARandomVariable(alpha)
    a = A.sample(100000)
    print(np.mean(a), A.mean())
    print(np.var(a), A.variance())

    x0 = A.laplace_transform(1)
    x1 = np.mean(np.exp(-a))

    assert np.isclose(x1, x0, rtol=1e-3), f"Expected {x1} to be close to {x0}."

