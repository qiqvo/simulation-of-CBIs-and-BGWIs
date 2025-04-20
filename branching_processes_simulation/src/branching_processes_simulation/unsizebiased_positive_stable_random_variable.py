from typing import Any, Callable
import numpy as np
from scipy.special import gamma, hyp1f1, gammaincc
from scipy.integrate import quad

from branching_processes_simulation.constant_variable import ConstantVariable
from branching_processes_simulation.positive_stable_random_variable import PositiveStableRandomVariable
from branching_processes_simulation.random_variable import RandomVariable


class UnsizebiasedPositiveStableRandomVariable(RandomVariable):
    def __new__(cls, alpha: float, d: float=1, *args, **kwargs):
        if alpha == 1:
            return ConstantVariable(d)
        return super().__new__(cls)
    
    # alpha < 1
    def __init__(self, alpha: float, d: float=1) -> None:
        assert alpha <= 1
        self.alpha = alpha
        self.d = d

        self._stable = PositiveStableRandomVariable(alpha, d)

    def characteristic_function(self, t: np.complex64) -> np.complex64:
        return self.laplace_transform(- 1j * t)

    def laplace_transform(self, t: np.float64) -> np.float64:
        return (1 - self.mean() * quad(lambda y: np.exp(-y**self.alpha), 0, t)[0])

    def pdf(self, x: np.float64) -> np.float64:
        return None # unknown

    def cdf(self, x: np.float64) -> np.float64:
        return None # unknown

    def mean(self) -> np.float64:
        if self.alpha < 1:
            return self.alpha / gamma(1/self.alpha)
        elif self.alpha == 1:
            return 1
        else:
            return None 

    def variance(self) -> np.float64:
        if self.alpha < 1:
            return np.infty
        else:
            return 0

    @staticmethod
    def _F(x, a, A):
        A = 1/A 

        term1_num = (
            A**(-1/a) * (1 + a) * x**((1 + 3 * a) / (2 * a - 2)) +
            A**((-a - 1)/a) * x**((5 * a + 1) / (2 * a - 2)) * a
        )
        term1 = term1_num * gamma((a - 1)/a) * hyp1f1(2, (4*a + 1)/a, x**(a / (a - 1)) / A) * a**5 * np.exp(-x**(a / (a - 1)) / A)
        term1 /= 12

        term2_inner1 = (
            (a + 0.5) * (1 + a) * A * x**((-1 - 3 * a) / (2 * a - 2)) +
            x**((-a - 1) / (2 * a - 2)) * a**2 / 2
        )
        upper_gamma_term = gamma((1 + 2 * a)/a) * gammaincc((1 + 2 * a)/a, x**(a / (a - 1)) / A)
        term2 = gamma((a - 1)/a) * term2_inner1 * a**2 * A * upper_gamma_term

        pi_term = (
            (a + 0.5) * A**2 * (1 + a) * x**((-1 - 3 * a) / (2 * a - 2)) +
            a**2 * (A * x**((-a - 1)/(2 * a - 2)) - np.sqrt(x)) / 2
        )
        term2 -= np.pi / np.sin(np.pi / a) * (1 + a) * pi_term
        term2 *= (a + 0.5) * A * (1/3 + a)

        numerator = 12 * (-term1 + term2)
        denominator = (
            np.sqrt(x) * gamma((a - 1)/a) * a**2 * (2 * a**2 + 3 * a + 1) * (1 + 3 * a) * A
        )

        return numerator / denominator * A**(1/a - 1)

    def _a(self, theta: np.ndarray[float]):
        if self.alpha > 0.99:
            sin_th = np.sin(theta)
            cot_th = np.sqrt(1 - sin_th**2)/sin_th
            a = 1 - self.alpha

            res = a*theta \
                - theta*(theta*cot_th + np.log(sin_th))*a**2 \
                + (-(2*theta**3)/3 + theta*(2*theta*cot_th \
                + np.log(sin_th) - 2)*np.log(np.sqrt(sin_th)))*a**3
        else:
            res = np.sin(self.alpha * theta)**self.alpha
            res /= np.sin(theta)
            res **= 1/self.alpha
            res = np.sin((1 - self.alpha) * theta) * res
        return res

    def sample(self, N: int, option='mcmc', **kwargs) -> np.ndarray[float]:
        alpha = self.alpha
        if option == 'cdf':
            theta, U = self.rng.uniform(0, 1, (2, N))
            A = self._a(theta * np.pi)

            # res = np.power(a / w, (1 - alpha) / alpha) * ((self.d)**(1/alpha))
            res = []
            return res
        elif option == 'mcmc':
            N_burn_in = kwargs.get('N_burn_in', 1000)
            X = self._stable.sample(N + 1 + N_burn_in, **kwargs)
            U = self.rng.uniform(0, 1, N + 1 + N_burn_in)
            for i in range(N + N_burn_in):
                if U[i] > X[i] / X[i + 1]:
                    X[i + 1] = X[i]
            return X[N_burn_in + 1:]