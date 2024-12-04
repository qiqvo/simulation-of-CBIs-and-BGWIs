import numpy as np

from branching_processes_simulation.continuous_space_process.cbi import CBI
from branching_processes_simulation.continuous_space_process.stable_cb import StableCB
from branching_processes_simulation.linnik import Linnik


class StableCBI(CBI):
    def __init__(self, alpha: np.float64, c: np.float64, d: np.float64) -> None:
        assert 0 < alpha <= 1 and d > 0 and c > 0
        super().__init__(lambda t: c * t**(1 + alpha), lambda t: -d * t**(alpha))        

        self.alpha = alpha
        self.c = c
        self.d = d
        self.delta = d / (alpha * c)
        self._cb = StableCB(alpha, c)
        self._linnik = Linnik(self.alpha, self.delta)

    def characteristic_function(self, t: np.complex64, time: np.float64, z: np.float64) -> np.complex64:
        return self.laplace_transform(-1j * t, time, z)

    def laplace_transform(self, t: np.float64, time: float, z: np.float64) -> np.float64:
        return (1 + self.alpha * self.c * np.abs(t)**self.alpha * time)**(-self.delta) * self._cb.laplace_transform(t, time, z)

    def mean(self, time: float, z: np.float64) -> np.float64:
        if self.alpha < 1:
            return np.infty
        else: 
            return z + self.delta * time

    def variance(self, time: float, z: np.float64) -> np.float64:
        if self.alpha < 1:
            return np.infty
        else: 
            return 2 * self.c

    def sample(self, N: int, time: float, z: np.float64) -> np.ndarray[float]:
        s = self._cb.sample(N, time, z)
        s = s + self._linnik.sample(N) * (self.alpha *self.c * time)**(1 / self.alpha)
        return s