# from matplotlib import pyplot as plt
import numpy as np
# import time
# import branching_processes_simulation as bps
# from branching_processes_simulation import linnik

import branching_processes_simulation_package

def main():
    print("OK.")
    # print(bps.linnik)
    X = linnik.Linnik(0.4, 0.5)
    print(X.sample(1))

if __name__ == '__main__':
    main()
