class Node():
    def __init__(self, parent=None):
        self.parent = parent
        self.children = []

    def create_offspring(self, N: int):
        for _ in range(N):
            self.children.append(Node(self))    

