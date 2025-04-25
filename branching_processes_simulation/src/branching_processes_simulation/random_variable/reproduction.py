import numpy as np

from branching_processes_simulation.random_variable.immigration_sl import ImmigrationSL, RandomVariable


class ReproductionSL(RandomVariable):
    def __init__(self, alpha, c, l, l_prime=None) -> None:
        super().__init__()
        self.alpha = alpha
        self.c = c
        self.l = l

        self._immigration_helper = None
        self.l_prime = l_prime

    def generating_function(self, s: np.complex64) -> np.complex64:
        return s + self.c * (1 - s)**(1 + self.alpha) * self.l(1 - s)

    def characteristic_function(self, t: np.complex64) -> np.complex64:
        return self.generating_function(np.exp(1j * t))

    def laplace_transform(self, t: np.float64) -> np.float64:
        return self.generating_function(np.exp(-t))

    def pdf(self, x: np.float64) -> np.float64:
        raise NotImplementedError()

    def cdf(self, x: np.float64) -> np.float64:
        raise NotImplementedError()
    
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
        immigration = ImmigrationSL(self.c * (1 + self.alpha), self.l1)
        return immigration

    def sample(self, N: int) -> np.ndarray[float]:
        if self._immigration_helper is None:
            self._immigration_helper = self._create_immigration()
        s = self._immigration_helper.sample(N) + 1
        s = self.rng.binomial(1, 1 / s) * s
        return s