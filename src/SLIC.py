import numpy as np
from cv2 import cv2
from mask import Mask
import pandas as pd


import time

class MaskedSLIC():
    def __init__(self, img, ROI, method=cv2.ximgproc.SLIC, region_size=50):

        superpixel = cv2.ximgproc.createSuperpixelSLIC(
            img, algorithm=method, region_size=region_size, ruler=30)
        superpixel.iterate()
        superpixel.enforceLabelConnectivity(25)
        contour_mask = superpixel.getLabelContourMask()
        contour_mask[ROI==0] = 0
        
        labels = superpixel.getLabels()
        labels[ROI==0] = -1


        self.labels_position = self.__get_labels_position(labels)
        self.__remap_labels(labels)
        self.labels = labels
        self.numOfPixel = np.max(labels) + 1
        self.contour_mask = contour_mask
        self.adjacent_pairs = self.__construct_adjacency(labels)


    def __get_labels_position(self,labels):
        data = labels.ravel()
        f = lambda x: np.unravel_index(x.index, labels.shape)
        temp = pd.Series(data).groupby(data).apply(f)
        temp = temp.reset_index(drop=True)
        return temp

    def __remap_labels(self, labels):
        for idx, (rows, cols) in enumerate(self.labels_position):
            labels[rows, cols] = idx

    def __construct_adjacency(self, labels):
        h,w = labels.shape[0:2]

        right_adjacent = np.zeros( ( (h-1),(w-1), 2), dtype=np.int32  )
        right_adjacent[:,:,0] = labels[0:h-1, 0:w-1]
        right_adjacent[:,:,1] = labels[0:h-1, 1:w]
        right_adjacent = right_adjacent.reshape((-1,2))

        bottom_adjacent = np.zeros( ( (h-1),(w-1), 2), dtype=np.int32   )
        bottom_adjacent[:,:,0] = labels[0:h-1, 0:w-1]
        bottom_adjacent[:,:,1] = labels[1:h, 0:w-1]
        bottom_adjacent = bottom_adjacent.reshape((-1,2))    


        adjacent_pairs = np.vstack((right_adjacent,bottom_adjacent))
        adjacent_pairs = np.unique(adjacent_pairs, axis=0)
        adjacent_pairs = adjacent_pairs[ np.invert( np.equal(adjacent_pairs[:,0], adjacent_pairs[:,1]) ) ]
        adjacent_pairs = np.vstack((adjacent_pairs,adjacent_pairs[:,-1::-1]))

        return adjacent_pairs


















