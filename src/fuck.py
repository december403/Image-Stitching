

import numpy as np
from cv2 import cv2
from matplotlib import pyplot as plt
import itertools as it
from func import *
import time



with open('GMS_matching_pair.npy', 'rb') as f:
    src_pts = np.load(f)
    dst_pts = np.load(f)

ref_img = cv2.imread('./DJI/DJI_0001.JPG')
tar_img = cv2.imread('./DJI/DJI_0002.JPG')

for x,y in src_pts:
    cv2.circle(tar_img, (int(x),int(y)), 3, (0,0,255), -1)  
cv2.imwrite('highlight_key_points.jpg', tar_img)