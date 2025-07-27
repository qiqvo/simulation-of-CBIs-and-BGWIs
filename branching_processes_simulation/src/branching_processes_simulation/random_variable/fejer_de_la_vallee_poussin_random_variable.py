import numpy as np

from branching_processes_simulation.random_variable.random_variable import (
    RandomVariable,
)


# Fejer-de la Vallee Poussin
class FejerDeLaValleePoussin(RandomVariable):
    _interval_a = -np.inf
    _interval_b = np.inf

    def __init__(self) -> None:
        return None

    def characteristic_function(self, t: np.complex64) -> np.complex64:
        return np.max(1 - np.abs(t), 0)

    def laplace_transform(self, t: np.float64) -> np.float64:
        raise ValueError(
            "Laplace transform is not defined for Fejer-de la Vallee Poussin distribution."
        )

    def pdf(self, x: np.float64) -> np.float64:
        return 1 / 2 / np.pi * (np.sin(x / 2) / (x / 2)) ** 2

    def cdf(self, x: np.float64) -> np.float64:
        return None

    def mean(self) -> np.float64:
        # Does not exist in Leb sense.
        return np.nan

    def absolute_moment(self, k: int) -> np.float64:
        return np.inf

    def variance(self) -> np.float64:
        return np.inf

    def sample(self, N: int) -> np.ndarray[float]:
        s = []
        i = 0
        while i < N:
            u = self.rng.uniform(-1, 1)
            v = self.rng.uniform(-1, 1)
            v_inv = 1 / v

            if u < 0:
                u, v, v_inv = -u * v * v, v_inv, v
            if u < np.sin(v_inv) ** 2:
                s.append(2 * v_inv)
                i += 1
        return np.array(s)
