import numpy as np
from scipy.stats import betaprime
from scipy import integrate

from branching_processes_simulation.random_variable import RandomVariable


class PolyaTransformedTau(RandomVariable):
    def __init__(self, alpha: float) -> None:
        self.alpha = alpha
        self._x = betaprime(self.alpha, self.alpha)

    def characteristic_function(self, t: np.complex64) -> np.complex64:
        return 1 - np.power(np.power(np.abs(t), self.alpha) / (1 + np.power(np.abs(t), self.alpha)),1/self.alpha)

    def laplace_transform(self, t: np.float64) -> np.float64:
        return np.real(self.characteristic_function(t))

    def pdf(self, x: np.float64) -> np.float64:
        xa = x**self.alpha
        return (1 + self.alpha) * xa / (1 + xa)**(2 + 1/self.alpha)

    def cdf(self, x: np.float64) -> np.float64:
        return x**(1 + self.alpha)/(1 + x**self.alpha)**((1 + self.alpha)/self.alpha)

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

    def sample(self, N: int) -> np.ndarray[float]:
        return self._x.rvs(N, random_state=self.rng)**(1/self.alpha)