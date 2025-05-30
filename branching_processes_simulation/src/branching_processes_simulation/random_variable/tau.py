from typing import Callable
import numpy as np

from branching_processes_simulation.random_variable.exponential import Exponential
from branching_processes_simulation.random_variable.linnik import Linnik
from branching_processes_simulation.random_variable.random_variable import RandomVariable
from branching_processes_simulation.random_variable.unsizebiased_positive_stable import UnsizebiasedPositiveStable


class Tau(RandomVariable):
    def __new__(cls, alpha: float, *args, **kwargs):
        if alpha == 1:
            return Exponential(1)
        return super().__new__(cls)

    def __init__(self, alpha: float) -> None:
        assert 0 < alpha <= 1

        self.alpha = alpha
        self._u_stable = UnsizebiasedPositiveStable(alpha)
        self._linnik_0 = Linnik(alpha, 1/alpha)
        self._linnik_1 = Linnik(alpha, 1 + 1/alpha)
        self._ber = lambda p, size: self.rng.binomial(n=1, p=p, size=size)

    def characteristic_function(self, t: np.complex64) -> np.complex64:
        return self.laplace_transform(-1j * t)

    def laplace_transform(self, t: np.float64) -> np.float64:
        return 1 - np.power(np.power(t, self.alpha) / (1 + np.power(t, self.alpha)), 1/self.alpha)

    def pdf(self, x: np.float64) -> np.float64:
        return None

    def cdf(self, x: np.float64) -> np.float64:
        return None

    def mean(self) -> np.float64:
        return 1

    def variance(self) -> np.float64:
        return np.infty

    def sample(self, N: int, option='cdf', **kwargs) -> np.ndarray[float]:
        if option == 'cdf':
            alpha = self.alpha
            X = self._u_stable.sample(N, **kwargs)
            U = self.rng.uniform(0, 1, N)

            return X * (-np.log(U))**(1/alpha)
        elif option == 'mcmc':
            N_burn_in = kwargs.get('N_burn_in', 1000)
            X = self._linnik_1.sample(N + 1 + N_burn_in, **kwargs)
            U = self.rng.uniform(0, 1, N + 1 + N_burn_in)
            for i in range(N + N_burn_in):
                if U[i] > X[i] / X[i + 1]:
                    X[i + 1] = X[i]
            return X[N_burn_in + 1:]
        raise ValueError(f"Unknown sampling option: {option}.")

    
    def function_expectation(self, theta: Callable, N=100, option='integrated_tail', theta_diff=None, **kwargs) -> np.ndarray[float]:
        if option == 'integrated_tail':
            delta = 0.001
            if theta_diff is None:
                theta_diff = lambda x: (theta(x + delta) - theta(x)) / delta
            res = theta(0) + self._linnik_0.sample_function(N, theta_diff, **kwargs).mean()
        elif option == 'size_biased_ber':
            s = self._linnik_1.sample(N)
            for i in range(N):
                h = self._ber(1/(s[i] + 1), 1)
                while h == 0:
                    s[i] = self._linnik_1.sample(1)[0]
                    h = self._ber(1/(s[i] + 1), 1)
                s[i] = theta(np.array([s[i]])) * (s[i] + 1) / s[i]
            res = s.mean()
        elif option == 'size_biased':
            s = self._linnik_1.sample(N)
            s = theta(s) / s
            res = s.mean()
        return res