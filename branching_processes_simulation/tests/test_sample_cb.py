from matplotlib import pyplot as plt
import numpy as np

from branching_processes_simulation.random_process.stable_cb import StableCB


def test_sample_cb():
    alpha = 0.6
    c = 1
    X = StableCB(alpha, c)

    t = 1
    z = 1
    s1 = X.sample(500000, t, [z])

    x = 1
    x0 = X.laplace_transform(x, t, z)
    x1 = (np.mean(np.exp(-x*s1)))

    assert np.isclose(x1, x0, rtol=0.001), f"Expected {x1} to be close to {x0}"
