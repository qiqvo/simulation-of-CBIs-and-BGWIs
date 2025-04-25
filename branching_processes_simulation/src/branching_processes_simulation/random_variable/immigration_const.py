import numpy as np

from branching_processes_simulation.random_variable.immigration_sl import ImmigrationSL


class ImmigrationConst(ImmigrationSL):
    def __init__(self, alpha, d) -> None:
        super().__init__(alpha, d, lambda x: 1)

    def sample(self, N: int) -> np.ndarray[float]:
        s = np.zeros(N)

        counter = 1
        k = self.rng.binom.rvs(N, self.d)
        s[:k] = counter
        while k > 0:
            k = self.rng.binom.rvs(k, 1 - self.alpha / counter)
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