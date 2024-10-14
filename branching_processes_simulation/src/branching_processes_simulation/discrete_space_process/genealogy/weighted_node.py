from typing import List
from branching_processes_simulation.discrete_space_process.genealogy.node import Node


class WeightedNode(Node):
    def __init__(self, parent=None, weight=0):
        super().__init__(parent)
        self.weight = weight

    def create_offspring(self, N: int):
        self.children.append(WeightedNode(self, N))