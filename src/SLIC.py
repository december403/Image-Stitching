import numpy as numpy
from cv2 import cv2

class SLIC():
    def __init__(img, mask, method=cv2.ximgproc.SLIC, region_size=50):
        img = cv2.cvtColor(img,cv2.COLOR_BGR2LAB)
        img = cv2.GaussianBlur(img,(5,5),0)
        superpixel = cv2.ximgproc.createSuperpixelSLIC(img,algorithm=cv2.ximgproc.SLIC, region_size=50)
        superpixel.iterate(10)
        superpixel.enforceLabelConnectivity(25)
        number = superpixel.getNumberOfSuperpixels()
        mask = superpixel.getLabelContourMask()
        label_mask = superpixel.getLabels()