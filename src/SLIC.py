import numpy as np
from cv2 import cv2
from mask import Mask
import time
from node import Node

class MaskedSLIC():
    def __init__(self, img, mask, method=cv2.ximgproc.SLIC, region_size=50):

        img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        img = cv2.GaussianBlur(img, (5, 5), 0)
        img[np.invert(mask>0)] = 0

        self.__superpixel = cv2.ximgproc.createSuperpixelSLIC(
            img, algorithm=method, region_size=region_size)
        self.__superpixel.iterate(10)
        self.__superpixel.enforceLabelConnectivity(25)

        numOfPixel = self.__superpixel.getNumberOfSuperpixels()
        contour_mask = self.__superpixel.getLabelContourMask()
        label = self.__superpixel.getLabels()

        masked_label = np.ones(label.shape) * (-1)
        masked_label[mask>0] = label[mask>0]

        masked_contour = np.copy(contour_mask)
        # masked_contour[mask==0] = 0

        new_number = 0
        for idx in range(numOfPixel):
            if np.any(masked_label==idx):
                masked_label[masked_label==idx] = new_number
                new_number += 1

        self.mask = masked_label
        self.numOfPixel = new_number
        self.masked_contour = masked_contour










#     tar_img[maskedSLIC.mask==idx] = (np.random.randint(0,256),np.random.randint(0,256),np.random.randint(0,256))
start_time = time.time()
tar_img = cv2.imread('./data/processed_image/warped_target.jpg')
ref_img = cv2.imread('./data/processed_image/warped_reference.jpg')
mask = Mask(tar_img, ref_img)
maskedSLIC = MaskedSLIC(tar_img, mask.overlap, region_size=100)

# for idx in range(maskedSLIC.numOfPixel):
#     tar_img[maskedSLIC.mask==idx] = (np.random.randint(0,256),np.random.randint(0,256),np.random.randint(0,256))

tar_img[maskedSLIC.masked_contour>0] = (0,255,0)
print(f'{time.time() - start_time} sec elapse')
cv2.imwrite('slic.png', tar_img)
if ref_img is None:
    print('fuck')

