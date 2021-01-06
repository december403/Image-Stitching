import numpy as np

class Vertex():
    def __init__(self, label, SLIC, mask, weight_map):
        self.label = label
        self.x_coordi = SLIC.labels_position[label][1]
        self.y_coordi = SLIC.labels_position[label][0]
        self.size = len(self.x_coordi)
        # self.adjacent_vertices = self.__find_adjcent_vertices(label, SLIC)
        self.is_on_tar_edge = self.__is_on_tar_edge(label, SLIC, mask)
        self.is_on_ref_edge = self.__is_on_ref_edge(label, SLIC, mask)
        self.weight = self.__calculate_weight(weight_map)

    def __is_on_tar_edge(self, label, SLIC, mask):
        tar_edge = mask.tar_overlap_edge
        return np.any( tar_edge[self.y_coordi, self.x_coordi] )



    def __is_on_ref_edge(self, label, SLIC, mask):
        ref_edge = mask.ref_overlap_edge
        return np.any( ref_edge[self.y_coordi, self.x_coordi] )



    # def __find_adjcent_vertices(self, label, SLIC):
    #     adjacent_pairs = SLIC.adjacent_pairs
    #     pairs = adjacent_pairs[ np.any(adjacent_pairs == self.label, axis=1) ]
    #     pairs = np.unique(pairs)
    #     adjacent_labels = np.delete( pairs, pairs==self.label )
    #     adjacent_labels = np.delete( adjacent_labels, adjacent_labels==0 ) 

    #     return adjacent_labels

    def __calculate_weight(self, weight_map):
        return np.sum(weight_map[self.y_coordi, self.x_coordi]) / self.size

