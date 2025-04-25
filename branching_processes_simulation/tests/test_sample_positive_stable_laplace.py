from matplotlib import pyplot as plt
import numpy as np
import time

from scipy.integrate import quad
from scipy.special import gamma

from branching_processes_simulation.random_variable.positive_stable import PositiveStable



def test_sample_positive_stable():
    alpha = 0.8
    S = PositiveStable(alpha)

    s1 = S.sample(100000, option='scipy')
    s2 = S.sample(100000, option='gen_CMS')
    s3 = S.sample(100000, option='CMS')

    x = 1
    x0 = S.laplace_transform(x)
    x1 = np.real(np.mean(np.exp(-x*s1)))
    x2 = np.real(np.mean(np.exp(-x*s2)))
    x3 = np.real(np.mean(np.exp(-x*s3)))

    print(x0, x1, x2, x3, sep='\n')
    eps = 0.005
    assert np.isclose(x1, x0, atol=0.0, rtol=eps), f"Expected {x1} to be close to {x0}."
    assert np.isclose(x2, x0, atol=0.0, rtol=eps), f"Expected {x2} to be close to {x0}."
    assert np.isclose(x3, x0, atol=0.0, rtol=eps), f"Expected {x3} to be close to {x0}."