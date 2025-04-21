from matplotlib import pyplot as plt
import numpy as np

from branching_processes_simulation.continuous_space_process.tau import Tau
from branching_processes_simulation.linnik import Linnik
from branching_processes_simulation.positive_stable_random_variable import PositiveStableRandomVariable


def test_tau():
    alpha = 0.6
    S = Tau(alpha)

    s1 = S.sample(500000)

    x = 1
    x0 = S.laplace_transform(x)
    x1 = np.mean(np.exp(-x*s1))

    assert np.isclose(x1, x0, rtol=0.001), f"Expected {x1} to be close to {x0}"
