import numpy as np


from branching_processes_simulation.random_variable.symmetric_stable import (
    SymmetricStable,
)


def test_sample_symm_stable():
    alpha = 0.8
    S = SymmetricStable(alpha)

    s1 = S.sample(100000, option="polya")
    s2 = S.sample(100000, option="scipy")
    s3 = S.sample(100000, option="CMS")

    x = 1
    x0 = np.exp(-(x**alpha))
    x1 = np.real(np.mean(np.exp(1j * x * s1)))
    x2 = np.real(np.mean(np.exp(1j * x * s2)))
    x3 = np.real(np.mean(np.exp(1j * x * s3)))

    print(x0, x1, x2, x3, sep="\n")
    assert np.isclose(
        x1, x0, atol=0.0, rtol=0.005
    ), f"Expected {x1} to be close to {x0}."
    assert np.isclose(
        x2, x0, atol=0.0, rtol=0.005
    ), f"Expected {x2} to be close to {x0}."
    assert np.isclose(
        x3, x0, atol=0.0, rtol=0.005
    ), f"Expected {x3} to be close to {x0}."
