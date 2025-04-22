from typing import List
from branching_processes_simulation.discrete_space_process.genealogy.node import Node
from branching_processes_simulation.discrete_space_process.genealogy.weighted_node import WeightedNode


class MarkedNode(Node):
    def __init__(self, parent=None, marks=0):
        super().__init__(parent)
        self.marks = marks

    def create_offspring(self, N: int, split_groups: List[int]):
        self.children.append(WeightedNode(self, N - len(split_groups)))
        for g in split_groups:
            self.children.append(MarkedNode(self, g))
