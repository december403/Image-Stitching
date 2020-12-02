
import numpy as np
from cv2 import cv2
from matplotlib import pyplot as plt
import itertools as it
from func import *
import time


start_time = time.time()
ref_img = cv2.imread('./DJI/DJI_0001.JPG')
tar_img = cv2.imread('./DJI/DJI_0002.JPG')
h, w, _ = ref_img.shape
h, w = tar_img.shape[0:2]
print(f'The reference image size is {ref_img.shape[0]}x{ref_img.shape[1]}')
print(f'The target image size is {h}x{w}')

with open('ORB_matching_pair.npy', 'rb') as f:
    src_pts = np.load(f)
    dst_pts = np.load(f)

H, _ = cv2.findHomography(src_pts,dst_pts)

warp_tar_img = cv2.warpPerspective(tar_img,H, tar_img.shape[0:2])
cv2.imwrite('warp_tar_img.jpg', warp_tar_img)