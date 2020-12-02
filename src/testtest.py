from cv2 import cv2 
import numpy as np
import os
print(os.path.isfile('src/lena_color.gif'))

img1 = cv2.imread('APAP_final.jpg')
print(type(img1))