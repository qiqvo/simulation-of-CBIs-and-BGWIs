from matplotlib import pyplot as plt
import numpy as np

from branching_processes_simulation.random_variable.tau import Tau
from branching_processes_simulation.random_variable.linnik import Linnik
from branching_processes_simulation.random_variable.positive_stable import (
    PositiveStable,
)


def test_tau_cdf():
    alpha = 0.3
    S = Tau(alpha)

    s1 = S.sample(500000)

    x = 1
    x0 = S.laplace_transform(x)
    x1 = np.mean(np.exp(-x * s1))

    assert np.isclose(
        x1, x0, atol=0.0, rtol=0.001
    ), f"Expected {x1} to be close to {x0}"
