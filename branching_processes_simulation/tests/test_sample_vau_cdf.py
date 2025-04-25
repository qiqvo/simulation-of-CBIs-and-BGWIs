import numpy as np

from branching_processes_simulation.random_variable.vau import Vau



def test_vau_sampling():
    alpha = 0.7

    V = Vau(alpha)
    a = V.sample(500000)
    print(np.mean(a), V.mean())
    print(np.var(a), V.variance())

    x = 1
    x0 = V.laplace_transform(x)
    x1 = np.mean(np.exp(-x * a))

    assert np.isclose(x1, x0, atol=0.0, rtol=0.005), f"Expected {x1} to be close to {x0}."

