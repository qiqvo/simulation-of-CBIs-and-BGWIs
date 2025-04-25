from concurrent.futures import ThreadPoolExecutor
import numpy as np
from scipy.integrate import quad


def parallel_integrate_upper_limits(func, a, bs):
    bs = [a] + sorted(bs)

    with ThreadPoolExecutor() as executor:
        res = executor.map(lambda i: quad(func, bs[i], bs[i+1])[0], range(len(bs)-1))
    res = list(res)

    return np.cumsum(res)