import numpy as np
from scipy.stats import bernoulli

from branching_processes_simulation.discrete_space_process.immigration_rv import ImmigrationRandomVariable, RandomVariable


class ReproductionRandomVariable(RandomVariable):
    def __init__(self, alpha, c, l) -> None:
        super().__init__()
        self.alpha = alpha
        self.c = c
        self.l = l
        self._immigration = None
        self.l_prime = None

    def generating_function(self, s: np.complex64) -> np.complex64:
        return s + self.c * (1 - s)**(1 + self.alpha) * self.l(1 - s)

    def characteristic_function(self, t: np.complex64) -> np.complex64:
        return self.generating_function(np.exp(1j * t))

    def laplace_transform(self, t: np.float64) -> np.float64:
        return np.real(self.generating_function(np.exp(-t)))

    def pdf(self, x: np.float64) -> np.float64:
        return None

    def cdf(self, x: np.float64) -> np.float64:
        return None
    
    def inverse_cdf(self, x: np.float64) -> np.float64:
        return None

    def mean(self) -> np.float64:
        return 1 

    # TODO: check
    def variance(self) -> np.float64:
        if self.alpha < 1:
            return np.infty
        else:
            return 2 * self.c * self.l(0)
        
    def _create_immigration(self):
        if self.l_prime is None:
            eps = 0.001
            self.l_prime = lambda x: (self.l(x + eps) - self.l(x)) / 0.001
        self.l1 = lambda x: self.l(x) + self.l_prime(x) * x / (1 + self.alpha)
        immigration = ImmigrationRandomVariable(self.c * (1 + self.alpha), self.l1)
        return immigration

    def sample(self, N: int) -> np.ndarray[float]:
        if self._immigration is None:
            self._immigration = self._create_immigration()
        s = self._immigration.sample(N) + 1
        s = bernoulli.rvs(1 / s) * s
        return s