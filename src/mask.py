from cv2 import cv2
import numpy as np

class Mask():
    def __init__(self, warp_tar_img, shift_ref_img):
        self.overlap = None
        self.tar = None
        self.ref = None
        # self.void = None
        self.tar_nonoverlap = None
        self.ref_nonoverlap = None
        self.constructMask(warp_tar_img,shift_ref_img)

    def constructMask(self, warped_tar_img, warped_ref_img):
        tar_img_gray = cv2.cvtColor(warped_tar_img,cv2.COLOR_BGR2GRAY)
        ref_img_gray = cv2.cvtColor(warped_ref_img,cv2.COLOR_BGR2GRAY)
        _, binary_tar_img = cv2.threshold(tar_img_gray,1,255,cv2.THRESH_BINARY)
        _, binary_ref_img = cv2.threshold(ref_img_gray,1,255,cv2.THRESH_BINARY)

        kernal = np.ones((5,5), np.int8)

        binary_tar_img = cv2.morphologyEx(binary_tar_img, cv2.MORPH_CLOSE, kernal)
        binary_ref_img = cv2.morphologyEx(binary_ref_img, cv2.MORPH_CLOSE, kernal)
        overlap = cv2.bitwise_and(binary_tar_img, binary_ref_img)

        # cv2.imwrite('binary_tar_img.jpg', binary_tar_img)
        # cv2.imwrite('binary_ref_img.jpg', binary_ref_img)
        # cv2.imwrite('overlap_img.jpg', overlap)

        self.overlap = overlap
        self.tar = binary_tar_img
        self.ref = binary_ref_img
        self.tar_nonoverlap = binary_tar_img - overlap
        self.ref_nonoverlap = binary_ref_img - overlap
        # self.void = np.ones()

        

