import numpy as np
from sympy import symbols, diff, Function

from branching_processes_simulation.random_variable import RandomVariable
from branching_processes_simulation.stable_random_variable import StableRandomVariable



class LinnikLaplaceTransform():
    def __init__(self, alpha: float, beta: float) -> None:
        x = symbols('x')
        self._f = 1 / (1 + x**alpha) ** beta
        self._x = x
        self._alpha = alpha
        self._beta = beta
        self._der_f_map = {0: self._f}

    def get_kth_derivative(self, k: int) -> Function:
        if k in self._der_f_map:
            return self._der_f_map[k]
        
        j = k-1
        while j not in self._der_f_map:
            --j
        
        der_f_k = diff(self._der_f_map[j], self._x, k - j)
        self._der_f_map[k] = der_f_k
        return der_f_k
    
    def get_kth_derivative_at_x(self, k: int, x: np.float64) -> np.float64:
        der_f_k = self.get_kth_derivative(k)
        return der_f_k.subs(self._x, x)


class Linnik(RandomVariable):
    def __init__(self, alpha: float, beta: float) -> None:
        self.alpha = alpha
        self.beta = beta
        # self._v = gengamma(self.delta, 1 / self.delta)
        self._s = StableRandomVariable(self.alpha)
        self._laplace_transform = LinnikLaplaceTransform(alpha, beta)

    def characteristic_function(self, t: np.complex64) -> np.complex64:
        return self._laplace_transform.get_kth_derivative_at_x(0, t)

    def laplace_transform(self, t: np.float64) -> np.float64:
        return self._laplace_transform.get_kth_derivative_at_x(0, t)

    def pdf(self, x: np.float64) -> np.float64:
        return None

    def cdf(self, x: np.float64) -> np.float64:
        return None
    
    def mean(self) -> np.float64:
        return np.inf if self.alpha < 1 else self.beta

    def variance(self) -> np.float64:
        return np.inf if self.alpha < 1 else self.beta

    def sample(self, N: int) -> np.ndarray[float]:
        v = self.rng.gamma(self.beta, 1, N)
        s = self._s.sample(N)
        return np.power(v, 1/self.alpha) * s
    
    def laplace_transform_kth_derivative(self, k: int, t: np.float64) -> np.float64:
        return (-1)**k * self._laplace_transform.get_kth_derivative_at_x(k, t)
