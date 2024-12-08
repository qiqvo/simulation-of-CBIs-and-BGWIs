from matplotlib import pyplot as plt
import numpy as np

from branching_processes_simulation.continuous_space_process.stable_cb import StableCB
from branching_processes_simulation.plotting.plot_stacked import plot_stacked_below

def test():
    alpha = 0.5
    c = 1
    time = 1

    cb = StableCB(alpha, c)
    cb_profile = cb.sample_profile(time, z=1)
    print(cb_profile)

    fig, c_profile = plot_stacked_below(time, cb_profile)
    fig.show()