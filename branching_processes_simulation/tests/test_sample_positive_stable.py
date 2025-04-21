from matplotlib import pyplot as plt
import numpy as np
import time

from scipy.integrate import quad
from scipy.special import gamma

from branching_processes_simulation.positive_stable_random_variable import PositiveStableRandomVariable


def test_positive_stable():
    alpha = 0.6
    S = PositiveStableRandomVariable(alpha)

    s1 = S.sample(500000, option='CMS')

    x = 1
    x0 = np.exp(-x**alpha)
    x1 = (np.mean(np.exp(-x*s1)))

    assert np.isclose(x1, x0, rtol=0.01), f"Expected {x1} to be close to {x0}"
