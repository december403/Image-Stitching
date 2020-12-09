import pandas as pd
import numpy as np
from mask import Mask
from cv2 import cv2 
import time

start = time.time()

tar_img = cv2.imread('./data/processed_image/warped_target.png')
ref_img = cv2.imread('./data/processed_image/warped_reference.png')
mask = Mask(tar_img, ref_img)
print('mask finishied')
tar_img[mask.overlap_edge >0] = (0,255,0)
print('render finishied')
cv2.imwrite('./overlap_edge.png',tar_img)

print(time.time() - start)