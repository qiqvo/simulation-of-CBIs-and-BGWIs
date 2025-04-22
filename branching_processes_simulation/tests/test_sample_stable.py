from matplotlib import pyplot as plt
import numpy as np
import time

from scipy.integrate import quad
from scipy.special import gamma

from branching_processes_simulation.random_variable.stable import Stable


def test_sample_stable():
    alpha = 0.55
    beta = -1
    S = Stable(alpha, beta)

    s2 = S.sample(500000, option='scipy')
    s3 = S.sample(500000, option='CMS')

    print(np.mean(np.abs(s2)**(alpha - 0.01)))
    print(np.mean(np.abs(s3)**(alpha - 0.01)))

    print(np.mean(np.abs(s2)**(-1)))
    print(np.mean(np.abs(s3)**(-1)))

    x = 0.01
    x0 = (S.characteristic_function(x))
    x2 = (np.mean(np.exp(1j * x*s2)))
    x3 = (np.mean(np.exp(1j * x*s3)))

    print(x0, x2, x3, sep='\n')
    # assert False
    assert np.isclose(x2, x0, rtol=0.01), f"Expected {x2} to be close to {x0}."
    assert np.isclose(x3, x0, rtol=0.01), f"Expected {x3} to be close to {x0}."