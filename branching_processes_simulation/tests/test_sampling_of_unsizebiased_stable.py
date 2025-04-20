from matplotlib import pyplot as plt
import numpy as np

from branching_processes_simulation.unsizebiased_positive_stable_random_variable import UnsizebiasedPositiveStableRandomVariable


def test():
    alpha = 0.6
    S = UnsizebiasedPositiveStableRandomVariable(alpha)
    N = 10 
    
    s = S.sample(N, option='MCMC')

    plt.scatter(range(N), s)
    plt.show()

    plt.hist(s, bins=20, density=True)
    plt.show()

# if __name__ == "__main__":
#     test()