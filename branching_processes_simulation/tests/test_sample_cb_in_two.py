from matplotlib import pyplot as plt
import numpy as np

from branching_processes_simulation.random_process.stable_cb import StableCB


def test_sample_cb_in_two():
    alpha = 0.6
    c = 1
    X = StableCB(alpha, c)

    t = 1
    z = 1
    N = 50000
    # s1 = X.sample(N, t, np.array([1, 2, 3, 4]) * z)

    s1 = X.sample(N, t / 2, [z])
    s2 = X.sample(1, t / 2, s1[0])
    # print(s2)
    s3 = X.sample(N, t, [z])
    times, s4 = X.sample_profile(N, t, z, t_per_1=2)
    s4 = s4[:, -1]

    x = 1
    x0 = X.laplace_transform(x, t, z)
    x2 = np.mean(np.exp(-x * s2))
    x3 = np.mean(np.exp(-x * s3))
    x4 = np.mean(np.exp(-x * s4))

    assert np.isclose(
        x2, x0, atol=0.0, rtol=0.005
    ), f"Expected {x2} to be close to {x0}"
    assert np.isclose(
        x3, x0, atol=0.0, rtol=0.005
    ), f"Expected {x3} to be close to {x0}"
    assert np.isclose(
        x4, x0, atol=0.0, rtol=0.005
    ), f"Expected {x2} to be close to {x0}"
