from abc import ABC, abstractmethod
from typing import Callable
import numpy as np

from branching_processes_simulation.settings import SEED

class IRandom(ABC):
    rng = np.random.default_rng(seed=SEED)