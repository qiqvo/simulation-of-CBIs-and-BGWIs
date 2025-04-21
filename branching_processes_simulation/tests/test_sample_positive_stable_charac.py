from matplotlib import pyplot as plt
import numpy as np
import time

from scipy.integrate import quad
from scipy.special import gamma

from branching_processes_simulation.positive_stable_random_variable import PositiveStableRandomVariable


def test_positive_stable():
    # alpha = 0.6
    # S = PositiveStableRandomVariable(alpha)

    # s1 = S.sample(1500000, option='CMS')
    # s2 = S.sample(1500000, option='gen_CMS')
    # s3 = S.sample(1500000, option='scipy')

    x = 1
    # x0 = np.exp(-(-1j * x)**alpha)
    # x1 = (np.mean(np.exp(1j * x * s1)))
    # x2 = (np.mean(np.exp(1j * x * s2)))
    # x3 = (np.mean(np.exp(1j * x * s3)))

    # print(np.exp(-x**alpha * (np.cos(np.pi * alpha / 2)) * (1 - 1j * np.tan(np.pi * alpha / 2))))

    # print(x0, x1, x2, x3, sep='\n')
    # print()
    # print(-x**alpha * (np.cos(np.pi * alpha / 2)) * (1 - 1j * np.tan(np.pi * alpha / 2)))
    # print(np.log(x0), np.log(x1), np.log(x2), np.log(x3), sep='\n')
    # print()
    # c = (np.log(x2) - np.log(x0)) / -x**alpha
    # print(x, c, np.abs(c))

    # As = np.linspace(0.05, 0.99, 50)
    # # xs = np.linspace(-20, 20, 50)
    # cs = []
    # # for x in xs:
    # for alpha in As:
    #     S = PositiveStableRandomVariable(alpha)
    #     s = S.sample(100000, option='scipy')
    #     x0 = np.exp(-(-1j * np.abs(x))**alpha)
    #     x1 = (np.mean(np.exp(1j * x * s)))

    #     c = (np.log(x1) - np.log(x0)) / -np.abs(x)**alpha
    #     cs.append(np.abs(c))
        
    # cs = np.array(cs)
    # # plt.plot(As, cs)
    # # plt.plot(As, np.arctan(cs / (1 - As)) *2 / np.pi)
    # # plt.plot(As, cs)
    # # plt.plot(As, (cs - 1) / (1 - As) / As**(-1/As))
    # # plt.plot(As, (cs / np.cos(np.pi * As / 2) / (gamma(1 - As) / As)))
    # # plt.plot(As, (cs / np.cos(np.pi * As / 2) * np.pi / 2))
    # # plt.plot(As, (np.cos(np.pi * As / 2)))
    # # plt.show()

    # # bs = (cs - 1) / (1 - As) / As**()
    # # plt.plot(As, np.log(bs))
    # # plt.show()


    # print(np.max(cs), As[np.argmax(cs)])

    # # assert np.isclose(x1, x0, rtol=0.01), f"Expected {x1} to be close to {x0}"
    # # assert np.isclose(x2, x0, rtol=0.01), f"Expected {x2} to be close to {x0}"
    # # assert np.isclose(x1 + 1, x0, rtol=0.01), f"Expected {x1} to be close to {x0}"
