# from matplotlib import pyplot as plt
import numpy as np
# import time
import branching_processes_simulation as bps
from branching_processes_simulation import linnik
from branching_processes_simulation.continuous_space_process import tau


# import branching_processes_simulation as 

def main():
    print("OK.")
    print(bps.linnik)
    X = linnik.Linnik(0.4, 0.5)
    print(X.sample(1))
    T = tau.Tau(0.4)
    print(T.sample(1))

if __name__ == '__main__':
    main()
