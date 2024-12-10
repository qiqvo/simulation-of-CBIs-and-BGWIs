from abc import ABC, abstractmethod
from typing import Callable
import numpy as np
from scipy.optimize import fsolve, root_scalar
from concurrent.futures import ThreadPoolExecutor

from branching_processes_simulation.i_random import IRandom

class RandomVariable(IRandom):
    _interval_a = -np.inf
    _interval_b = +np.inf

    def __init__(self) -> None:
        self._table = None 

    @abstractmethod
    def characteristic_function(self, t: np.complex64) -> np.complex64:
        return None

    @abstractmethod
    def laplace_transform(self, t: np.float64) -> np.float64:
        return None

    @abstractmethod
    def pdf(self, x: np.float64) -> np.float64:
        return None

    @abstractmethod
    def cdf(self, x: np.float64) -> np.float64:
        return None
    
    def inverse_cdf(self, x: np.float64) -> np.float64:
        return root_scalar(lambda t: self.cdf(t) - x, bracket=(self._interval_a, self._interval_b))[0]

    @abstractmethod
    def mean(self) -> np.float64:
        return None

    @abstractmethod
    def variance(self) -> np.float64:
        return None

    @abstractmethod
    def sample(self, N: int, **kwargs) -> np.ndarray[float]:
        return None

    def sample_function(self, N: int, theta: Callable, **kwargs) -> np.ndarray[float]:
        s = self.sample(N)
        return theta(s)
    
    def function_expectation(self, theta: Callable, N=100, **kwargs) -> np.ndarray[float]:
        return self.sample_function(N, theta).mean()
    
    def _choose_x0(self):
        if self._interval_a != -np.inf:
            x0 = self._interval_a
        elif self._interval_b != np.inf:
            x0 = self._interval_b
        else:
            x0 = 0
        return x0

    def precompute_cdf_table(self, granularity=100, x0=None):
        if x0 is None:
            x0 = self._choose_x0()

        us = np.linspace(0, 1, granularity, True)[1:-1]
        self._table = np.zeros(len(us)+2)
        def solve(u, x0, cdf, pdf):
            return root_scalar(
                lambda x: cdf(x) - u, x0=x0, 
                # fprime=pdf # if u > 0.1 else None
                ).root
        
        self._table[0] = self._interval_a
        self._table[-1] = self._interval_b

        with ThreadPoolExecutor() as executor: 
            res = executor.map(lambda u_i: solve(u_i, x0, self.cdf, self.pdf), us)
        
        res = list(res)
        self._table[1:-1] = res
        
    def sample_from_cdf(self, N: int, pdf_available=False, **kwargs) -> np.ndarray[float]:
        if self._table is None or (self._table and len(self._table) < 2 * N and N < 1e6):
            self.precompute_cdf_table(1000) # int(min(N, 1e6)))
    
        us = self.rng.uniform(0, 1, N)
        # inds = (us * (len(self._table) - 1)).astype(int)
        
        def solve(u, bracket, x0, cdf, pdf):
            if x0 == -np.inf:
                x0 = bracket[1]
            if bracket[1] == np.inf or bracket[0] == -np.inf:
                bracket = None
            
            return root_scalar(
                lambda x: cdf(x) - u, 
                bracket=bracket,
                x0=x0,
                fprime=pdf).root
        
        def parallel_solve(u):
            i = int(u * (len(self._table) - 1))
            l, r = i, i+1
            while self.cdf(self._table[l]) > u:
                l -= 1
            while self.cdf(self._table[r]) < u:
                r += 1

            return solve(u, 
                    bracket=(self._table[l], self._table[r]), 
                    x0=self._table[i],
                    cdf=self.cdf, 
                    pdf=self.pdf if pdf_available else None)

        with ThreadPoolExecutor() as executor:
            cs = executor.map(parallel_solve, us)

        return np.array(list(cs))