import numpy as np

class Node():
    def __init__(self, idx, x_y_coordi):
        self.x_lst = np.array(x_y_coordi[1])
        self.y_lst = np.array(x_y_coordi[0])
        self.idx= idx
        self.neighbors = []
        self.onEdge = False
        self.label = None