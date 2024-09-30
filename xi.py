import numpy as np

from fejer_de_la_vallee_poussin_random_variable import FejerDeLaValleePoussinRandomVariable
from polya_transformed_xi import PolyaTransformedXi
from random_variable import RandomVariable


class Xi(RandomVariable):
    def __init__(self, alpha: float) -> None:
        self.alpha = alpha
        self._fvp = FejerDeLaValleePoussinRandomVariable()
        self._pxi = PolyaTransformedXi(alpha)

    def characteristic_function(self, t: np.complex64) -> np.complex64:
        return 1 - np.power(np.power(np.abs(t), self.alpha) / (1 + np.power(np.abs(t), self.alpha)),1/self.alpha)

    def laplace_transform(self, t: np.float64) -> np.float64:
        return np.real(self.characteristic_function(t))

    def pdf(self, x: np.float64) -> np.float64:
        return None

    def cdf(self, x: np.float64) -> np.float64:
        return None

    #TODO: check
    def mean(self) -> np.float64:
        return np.infty

    #TODO: check
    def variance(self) -> np.float64:
        return np.infty

    def sample(self, N: int) -> np.ndarray[float]:
        s = np.abs(self._fvp.sample(N)) / self._pxi.sample(N)
        return s