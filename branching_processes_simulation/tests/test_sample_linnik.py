import numpy as np


from branching_processes_simulation.random_variable.linnik import Linnik


def test_linnik():
    alpha = 0.6
    beta = 1.7
    S = Linnik(alpha, beta)

    s1 = S.sample(500000)

    x = 1
    x0 = (1 + x**alpha) ** (-beta)
    x1 = np.mean(np.exp(-x * s1))

    assert np.isclose(
        x1, x0, atol=0.0, rtol=0.005
    ), f"Expected {x1} to be close to {x0}"
