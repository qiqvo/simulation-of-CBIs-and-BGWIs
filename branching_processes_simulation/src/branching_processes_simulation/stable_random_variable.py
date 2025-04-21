from typing import Any, Callable
import numpy as np
import scipy

from branching_processes_simulation.random_variable import RandomVariable


class StableRandomVariable(RandomVariable):
    """
    We use Zolotorev's representation of the stable distribution:
    phi(t) = exp(-d * |t|^alpha * exp(-i * pi * alpha / 2 * sign(t) * beta)).
    """

    def __init__(self, alpha: float, beta:float, d: float=1) -> None:
        ## alpha > 1 is not supported
        assert 0 < alpha < 1 and d > 0 and -1 <= beta <= 1

        self.alpha = alpha
        self.d = d
        self.beta = beta
        
        ## scipy uses a different parameterization:
        beta_scipy = np.tan(np.pi * alpha / 2 * beta) / np.tan(np.pi * alpha / 2)
        # TODO: check the scaling! 
        self._s = scipy.stats.levy_stable(alpha=alpha, beta=beta_scipy, loc=0, 
                                          scale=d * np.cos(np.pi * alpha / 2 * beta))
        self._s.random_state = self.rng

    def characteristic_function(self, t: np.float64) -> np.complex64:
        if t == 0: 
            return 1
        
        a1 = 1
        if self.beta != 0:
            re_t = np.real(t)
            sign_t = re_t / np.abs(re_t)

            a1 = np.exp(-1j * np.pi * self.alpha / 2 * sign_t * self.beta)
            
        return np.exp(-self.d * np.power(np.abs(t), self.alpha) * a1)

    def laplace_transform(self, t: np.float64) -> np.float64:
        return np.real(self.characteristic_function(-t * 1j))

    def pdf(self, x: np.float64) -> np.float64:
        raise NotImplementedError()

    def cdf(self, x: np.float64) -> np.float64:
        raise NotImplementedError()

    def mean(self) -> np.float64:
        return np.inf if self.beta == 1 else -np.inf if self.beta == -1 else np.nan

    def variance(self) -> np.float64:
        return np.inf
    
    @staticmethod
    def K(alpha: float) -> float:
        return 1 - np.abs(1 - alpha)

    # corresponds to the CMS paper notation. Not Zolotorev's.
    @staticmethod
    def stable_a(alpha, beta, theta: np.ndarray[float]):
        res = np.cos((1 - alpha) * theta - np.pi / 2 * beta * StableRandomVariable.K(alpha))
        res /= np.cos(theta)
        res **= (1 - alpha) / alpha
        res *= np.sin(alpha * theta + np.pi / 2 * beta * StableRandomVariable.K(alpha))
        res /= np.cos(theta)
        # res **= alpha/(1 - alpha)
        return res
    
    def sample(self, N: int, option='scipy', **kwargs) -> np.ndarray[float]:
        alpha = self.alpha
        if option == 'scipy':
            res = self._s.rvs(size=N)
        elif option == 'CMS':
            Phi = self.rng.uniform(-np.pi / 2, np.pi / 2, N)
            W = -np.log(self.rng.uniform(0, 1, N))
            res = self.stable_a(alpha, self.beta, Phi) / W**((1 - alpha) / alpha)

            res *= self.d ** 1/self.alpha
        return res