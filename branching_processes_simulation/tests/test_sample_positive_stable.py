import numpy as np


from branching_processes_simulation.random_variable.positive_stable import (
    PositiveStable,
)


def test_positive_stable():
    alpha = 0.6
    S = PositiveStable(alpha)

    s1 = S.sample(500000, option="CMS")

    x = 1
    x0 = np.exp(-(x**alpha))
    x1 = np.mean(np.exp(-x * s1))

    assert np.isclose(
        x1, x0, atol=0.0, rtol=0.005
    ), f"Expected {x1} to be close to {x0}"
