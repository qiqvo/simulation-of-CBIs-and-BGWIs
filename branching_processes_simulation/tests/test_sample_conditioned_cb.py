from matplotlib import pyplot as plt
import numpy as np

from branching_processes_simulation.random_process.conditioned_stable_cb import (
    ConditionedStableCB,
)


def test_sample_conditioned_cb():
    alpha = 0.6
    c = 1
    X = ConditionedStableCB(alpha, c)

    t = 1
    z = 1
    s1 = X.sample(500000, t, [z])
    # s2 = X.sample(500000, t, [z], option='poisson')

    x = 1
    x0 = X.laplace_transform(x, t, z)
    x1 = np.mean(np.exp(-x * s1))
    # x2 = (np.mean(np.exp(-x*s2)))

    assert np.isclose(
        x1, x0, atol=0.0, rtol=0.005
    ), f"Expected {x1} to be close to {x0}"
    # assert np.isclose(x2, x0, atol=0.0, rtol=0.001), f"Expected {x2} to be close to {x0}"
