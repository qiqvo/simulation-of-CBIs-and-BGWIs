from matplotlib import pyplot as plt
import numpy as np
import time

from scipy.integrate import quad
from scipy.special import gamma

from branching_processes_simulation.positive_stable_random_variable import PositiveStableRandomVariable


def test_positive_stable():
    alpha = 0.6
    S = PositiveStableRandomVariable(alpha)

    s1 = S.sample(1500000, option='CMS')
    s2 = S.sample(1500000, option='gen_CMS')
    s3 = S.sample(1500000, option='scipy')

    x = 1
    x0 = np.exp(-x**alpha)
    x1 = (np.mean(np.exp(-x*s1)))
    x2 = (np.mean(np.exp(-x*s2)))
    x3 = (np.mean(np.exp(-x*s3)))

    print(s2[s2 < 0])
    print(s3[s3 < 0])

    print(x0, x1, x2, x3)

    As = np.linspace(0.05, 0.99, 50)
    cs = []
    for a in As:
        S = PositiveStableRandomVariable(a)
        s = S.sample(500000, option='scipy')
        x0 = np.exp(-x**a)
        x1 = (np.mean(np.exp(-x*s)))

        c = np.log(x1)/np.log(x0)
        cs.append(c)
        
    cs = np.array(cs)
    # plt.plot(As, np.arctan(cs / (1 - As)) *2 / np.pi)
    plt.plot(As, cs)
    plt.plot(As, (cs - 1) / (1 - As) / As**(-1/As))
    # plt.plot(As, (cs / np.cos(np.pi * As / 2) / (gamma(1 - As) / As)))
    # plt.plot(As, (cs / np.cos(np.pi * As / 2) * np.pi / 2))
    # plt.plot(As, (np.cos(np.pi * As / 2)))
    plt.show()

    # bs = (cs - 1) / (1 - As) / As**()
    # plt.plot(As, np.log(bs))
    # plt.show()


    print(np.max(cs), As[np.argmax(cs)])

    assert np.isclose(x1, x0, rtol=0.01), f"Expected {x1} to be close to {x0}"
    assert np.isclose(x2, x0, rtol=0.01), f"Expected {x2} to be close to {x0}"
    assert np.isclose(x3, x0, rtol=0.01), f"Expected {x3} to be close to {x0}"
