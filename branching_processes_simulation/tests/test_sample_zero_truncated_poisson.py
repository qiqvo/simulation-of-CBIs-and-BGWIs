import numpy as np

from branching_processes_simulation.random_variable.zero_truncated_poisson import (
    ZeroTruncatedPoisson,
)


def test_sample_zero_truncated_poisson():
    rate = 6
    X = ZeroTruncatedPoisson(rate)

    s1 = X.sample(500000, option="cdf")
    s2 = X.sample(500000, option="poisson")

    x = 1
    x0 = X.laplace_transform(x)
    x1 = np.mean(np.exp(-x * s1))
    x2 = np.mean(np.exp(-x * s2))

    assert np.isclose(
        x1, x2, atol=0.0, rtol=0.005
    ), f"Expected {x1} to be close to {x0}"
    assert np.isclose(
        x1, x0, atol=0.0, rtol=0.005
    ), f"Expected {x1} to be close to {x0}"
    assert np.isclose(
        x2, x0, atol=0.0, rtol=0.005
    ), f"Expected {x2} to be close to {x0}"
