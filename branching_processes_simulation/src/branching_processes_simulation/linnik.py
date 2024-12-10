import numpy as np
from sympy import symbols, diff, Function

from branching_processes_simulation.continuous_space_process.fejer_de_la_vallee_poussin_random_variable import FejerDeLaValleePoussinRandomVariable
from branching_processes_simulation.exponential import Exponential
from branching_processes_simulation.random_variable import RandomVariable
from branching_processes_simulation.stable_random_variable import PositiveStableRandomVariable



class LinnikLaplaceTransform():
    def __init__(self, alpha: float, beta: float) -> None:
        x = symbols('x')
        self._f = 1 / (1 + x**alpha) ** beta
        self._x = x
        self._alpha = alpha
        self._beta = beta
        self._der_f_map = {0: self._f}

    def value(self, x: np.float64) -> np.float64:
        return self._der_f_map[0].subs(self._x, x)

    def get_kth_derivative(self, k: int) -> Function:
        if k in self._der_f_map:
            return self._der_f_map[k]
        
        j = k-1
        while j not in self._der_f_map and j >= 0:
            j -= 1
        
        der_f_k = diff(self._der_f_map[j], self._x, k - j)
        self._der_f_map[k] = der_f_k
        return lambda x: der_f_k.subs(self._x, x)
    
    def get_kth_derivative_at_x(self, k: int, x: np.float64) -> np.float64:
        der_f_k = self.get_kth_derivative(k)
        return der_f_k(x)


class Linnik(RandomVariable):
    def __new__(cls, alpha: float, *args, **kwargs):
        if alpha == 1:
            return Exponential(1)
        return super().__new__(cls)
    
    def __init__(self, alpha: float, beta: float) -> None:
        assert 0 < alpha <= 1 and beta > 0
            
        self.alpha = alpha
        self.beta = beta
        # self._v = gengamma(self.delta, 1 / self.delta)
        self._fvp = FejerDeLaValleePoussinRandomVariable()
        self._s = PositiveStableRandomVariable(self.alpha)
        self._laplace_transform = LinnikLaplaceTransform(alpha, beta)


    def characteristic_function(self, t: np.complex64) -> np.complex64:
        return self._laplace_transform.get_kth_derivative_at_x(0, np.abs(t))

    def laplace_transform(self, t: np.float64) -> np.float64:
        return self._laplace_transform.get_kth_derivative_at_x(0, t)

    def pdf(self, x: np.float64) -> np.float64:
        return None # unknown

    def cdf(self, x: np.float64) -> np.float64:
        return None # unknown
    
    def mean(self) -> np.float64:
        return np.inf if self.alpha < 1 else self.beta

    def variance(self) -> np.float64:
        return np.inf if self.alpha < 1 else self.beta

    def sample(self, N: int, option='stable') -> np.ndarray[float]:
        alpha = self.alpha
        if option=='stable':
            v = self.rng.gamma(self.beta, 1, N)
            s = self._s.sample(N)
            return np.power(v, 1/alpha) * s
        elif option == 'polya':
            v = np.abs(self._fvp.sample(N))
            u = self.rng.uniform(0, 1, N)
            z = ((1+alpha - np.sqrt((1+alpha)**2 - 4 * alpha * u)) / (2 * u) - 1)**(1/alpha)
            return v/z
            
    def laplace_transform_kth_derivative_at_x(self, k: int, t: np.float64) -> np.float64:
        # TODO: (-1)**k * ?
        return self._laplace_transform.get_kth_derivative_at_x(k, t)

    def laplace_transform_kth_derivative(self, k: int) -> Function:
        return self._laplace_transform.get_kth_derivative(k)
