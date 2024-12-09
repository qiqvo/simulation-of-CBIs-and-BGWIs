import numpy as np
from scipy.stats import betaprime
from scipy import integrate
from scipy.optimize import fsolve

from concurrent.futures import ThreadPoolExecutor

from branching_processes_simulation.random_variable import RandomVariable


class PolyaTransformedTau(RandomVariable):
    def __init__(self, alpha: float) -> None:
        self.alpha = alpha
        self._x = betaprime(self.alpha, self.alpha)

    def characteristic_function(self, t: np.complex64) -> np.complex64:
        return None

    def laplace_transform(self, t: np.float64) -> np.float64:
        return None # unknown

    def pdf(self, y: np.float64) -> np.float64:
        alpha = self.alpha
        ca = np.cos(np.pi * alpha / 2)
        abs_ca = np.abs(ca)

        numerator = 4 * ((y**(2*alpha)*alpha + (-2*alpha - 1)*y**(4*alpha)) * abs_ca**5 +
                        ((((-alpha + 1)*y**(2*alpha) + 3*y**(4*alpha)*(alpha + 1))*ca**2)/2 +
                        (1/2 - alpha)*y**(2*alpha) + y**(4*alpha)*(alpha + 1)) * abs_ca**3 +
                        4*((-alpha + 3/8)*y**(3*alpha) + ((alpha + 1)*y**(5*alpha))/16 +
                            y**alpha*((y**alpha)**2*alpha - alpha/16 + 1/16)) * ca**4) * alpha
        denominator = ((y**(2*alpha) + 1) * abs_ca + 2 * ca**2 * y**alpha)**3 * y
        return numerator / denominator

    def cdf(self, y: np.float64) -> np.float64:
        alpha = self.alpha
        ca = np.cos(np.pi * alpha / 2)
        abs_ca = np.abs(ca)

        numerator = (
            ((-alpha + 3)*y**(3*alpha) - y**alpha*(alpha - 1))*abs_ca**3 -
            2*ca**2*(-ca**2 * y**(2*alpha) + (-1/2 + alpha)*y**(2*alpha) - y**(4*alpha)/2)
        )
        denominator = (
            (abs_ca + y**(2*alpha)*abs_ca + 2*ca**2 * y**alpha)**2
        )
        return numerator / denominator

    def mean(self) -> np.float64:
        return np.infty

    def variance(self) -> np.float64:
        return np.infty
    
    def fractional_moment(self, beta:float) -> np.float64:
        if beta >= self.alpha: 
            return np.inf
        else:
            p1 = integrate.quad(lambda x: self.pdf(x) * x**beta, 0, 1)[0]
            p2 = integrate.quad(
                lambda x: x**(- beta/self.alpha) / (1 + x)**(2 + 1/self.alpha), 
                0, 1
            )[0]
            return p1 + (1 + self.alpha)/self.alpha * p2 

    def sample(self, N: int, option='cdf', **kwargs) -> np.ndarray[float]:
        if option=='rejection_sampling':
            return None
        elif option == 'cdf':
            def solve(u, cdf, pdf):
                return fsolve(lambda x: cdf(x) - u, x0=1, fprime=pdf)[0]

            u = self.rng.uniform(0, 1, N)
            with ThreadPoolExecutor() as executor:  # You can also use ProcessPoolExecutor
                u_values = list(executor.map(lambda u_i: solve(u_i, self.cdf, self.pdf), u))
            return np.array(u_values)
        return None