from matplotlib import pyplot as plt
import numpy as np
import time

from scipy.integrate import quad
from scipy.special import gamma

from branching_processes_simulation.random_variable.linnik import Linnik
from branching_processes_simulation.random_variable.positive_stable import PositiveStable


def test_linnik():
    alpha = 0.6
    beta = 1.7
    S = Linnik(alpha, beta)

    s1 = S.sample(500000)

    x = 1
    x0 = (1 + x**alpha)**(-beta)
    x1 = (np.mean(np.exp(-x*s1)))

    assert np.isclose(x1, x0, rtol=0.005), f"Expected {x1} to be close to {x0}"
