import numpy as np
from scipy.stats import betaprime
from scipy import integrate
from scipy.optimize import fsolve

from concurrent.futures import ThreadPoolExecutor

from branching_processes_simulation.random_variable import RandomVariable


class PolyaTransformedTau(RandomVariable):
    _interval_a = 0
    # _interval_b = +np.inf

    def __init__(self, alpha: float) -> None:
        super().__init__()
        self.alpha = alpha
        self._x = betaprime(self.alpha, self.alpha)

    def characteristic_function(self, t: np.complex64) -> np.complex64:
        return None

    def laplace_transform(self, t: np.float64) -> np.float64:
        return None # unknown

    def pdf(self, y: np.float64) -> np.float64:
        y = np.asarray(y)
        alpha = self.alpha
        ca = np.cos(np.pi * alpha / 2)
        abs_ca = np.abs(ca)

        numerator = 4 * y**(1-alpha) * ((y**(2*alpha)*alpha + (-2*alpha - 1)*y**(4*alpha)) * abs_ca**5 +
                        ((((-alpha + 1)*y**(2*alpha) + 3*y**(4*alpha)*(alpha + 1))*ca**2)/2 +
                        (1/2 - alpha)*y**(2*alpha) + y**(4*alpha)*(alpha + 1)) * abs_ca**3 +
                        4*((-alpha + 3/8)*y**(3*alpha) + ((alpha + 1)*y**(5*alpha))/16 +
                            y**alpha*((y**alpha)**2*alpha - alpha/16 + 1/16)) * ca**4) * alpha
        denominator = ((y**(2*alpha) + 1) * abs_ca + 2 * ca**2 * y**alpha)**3 * y
        res = numerator / denominator / y**(1-alpha)
        res = np.where(y <= 0, 0, res)
        return res if res.shape else res.item()

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

    def sample(self, N: int, option='cdf', **kwargs) -> np.ndarray[float]:
        if option=='rejection_sampling':
            return None
        elif option == 'cdf':
            return self.sample_from_cdf(N)
        return None
    
    def inverse_tail_cdf(self, x):
        alpha = self.alpha
        ca = np.cos(np.pi * alpha / 2)
        abs_ca = np.abs(ca)

        p = [
                abs_ca**2*x,
                (-abs_ca**3*alpha + 4*x*ca**2*abs_ca - abs_ca**3),
                (4*x*ca**4 - 2*ca**4 - 2*alpha*ca**2 + 2*abs_ca**2*x - ca**2),
                (-abs_ca**3*alpha + 4*x*ca**2*abs_ca - 3*abs_ca**3),
                abs_ca**2*x - ca**2
            ]
        
        res = np.roots(p)
        for r in res:
            if r.imag == 0 and r.real > 0:
                return r.real**(1/alpha)
        raise Exception('Undefinied behaviour!')

    def sample_from_cdf(self, N, **kwargs):
        u = self.rng.uniform(0, 1, N)

        with ThreadPoolExecutor() as executor:  
            cs = executor.map(self.inverse_tail_cdf, u)

        return np.array(list(cs))
