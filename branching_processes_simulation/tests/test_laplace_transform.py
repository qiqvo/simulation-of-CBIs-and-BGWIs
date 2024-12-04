from matplotlib import pyplot as plt
import numpy as np
import time

from branching_processes_simulation.discrete_space_process.bgw import BGW
from branching_processes_simulation.discrete_space_process.bgwi import BGWI
from branching_processes_simulation.discrete_space_process.immigration_exp_rv import ImmigrationExpRandomVariable
from branching_processes_simulation.discrete_space_process.reproduction_exp_rv import ReproductionExpRandomVariable
from branching_processes_simulation.linnik import Linnik

def test():
    alpha = 0.4
    c = 0.4
    d = 0.2
    delta = d / (alpha * c)

    L = Linnik(alpha, delta)
    print(L.sample(1))
    # print(L.laplace_transform_kth_derivative_at_x(1, 1))
    print(L.laplace_transform_kth_derivative_at_x(4, 1))
    
if __name__ == "__main__":
    test()