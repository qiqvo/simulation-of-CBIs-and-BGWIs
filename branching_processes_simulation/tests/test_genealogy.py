from matplotlib import pyplot as plt
import numpy as np
import time
from anytree import RenderTree

from branching_processes_simulation.discrete_space_process.bgw import BGW
from branching_processes_simulation.discrete_space_process.bgwi import BGWI
from branching_processes_simulation.discrete_space_process.immigration_exp_rv import ImmigrationExpRandomVariable
from branching_processes_simulation.discrete_space_process.reproduction_exp_rv import ReproductionExpRandomVariable

def dontrun():
    alpha = 0.4
    c = 0.4
    d = 0.2
    delta = d / (alpha * c)

    # L = Linnik(alpha, delta)
    # print(L.sample(1))

    xi = ReproductionExpRandomVariable(alpha, c)
    X = BGW(xi)

    time = 10
    z = 3
    # print()
    # print(X.sample_profile(time, z))

    # genealogy = X.sample_genealogy(time, z)
    # print(genealogy)
    # print()
    # for pre, _, node in RenderTree(genealogy):
    #     print(f"{pre}node")

    # x_profile = X.sample_profile_from_genealogy(time, genealogy)
    # print()
    # print(x_profile)


    eta = ImmigrationExpRandomVariable(alpha, d)
    Z = BGWI(xi, eta)
    # print()
    # print(Z.sample_profile(time, z))
    genealogy = Z.sample_genealogy(time, z)
    # print(genealogy)
    print()
    for pre, _, node in RenderTree(genealogy):
        print(f"{pre}node")

    z_profile = Z.sample_profile_from_genealogy(time, genealogy)
    print()
    print(z_profile)

# if __name__ == '__main__':
#     main()
