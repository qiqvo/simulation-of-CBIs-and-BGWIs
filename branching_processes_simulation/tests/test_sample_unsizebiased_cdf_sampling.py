import numpy as np

from branching_processes_simulation.random_variable.unsizebiased_positive_stable import UnsizebiasedPositiveStable


def test_unsizebiased_cdf_sampling():
    alpha = 0.6
    S = UnsizebiasedPositiveStable(alpha)

    s = S.sample(500000, option='cdf')
    # plt.hist(s, bins=100)
    # plt.show()

    C = S.mean()

    print(np.mean(s), S.mean())
    print(np.std(s))

    x = 1
    x1 = S.laplace_transform(x)
    x2 = (np.mean(np.exp(-x*s)))
    assert np.isclose(x2, x1, atol=0.0, rtol=0.005), f"Expected {x1} to be close to {x2}"
