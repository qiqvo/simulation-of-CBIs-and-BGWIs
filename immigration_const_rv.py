import numpy as np
from scipy.stats import bernoulli, binom

from immigration_rv import ImmigrationRandomVariable


class ImmigrationConstRandomVariable(ImmigrationRandomVariable):
    def __init__(self, alpha, d) -> None:
        super().__init__(alpha, d, lambda x: 1)
        self._b = bernoulli(self.d)

    def sample(self, N: int) -> np.ndarray[float]:
        # s = self._b.rvs(N)

        s = np.zeros(N)

        counter = 1
        k = binom.rvs(N, self.d)
        s[:k] = counter
        while k > 0:
            k = binom.rvs(k, 1 - self.alpha / counter)
            s[:k] += 1
            counter += 1
        np.random.shuffle(s)

        # counter = 1
        # k = len(s[s==counter])
        # while k > 0:
        #     s[s==counter] += bernoulli.rvs(1 - self.alpha / counter, size=k)
        #     counter += 1
        #     k = len(s[s==counter])

        # for i in range(N):
        #     counter = 0
        #     while s[i] != 0:
        #         counter += 1
        #         s[i] *= bernoulli.rvs(1 - self.alpha / counter)
        #     s[i] = counter
        
        return s