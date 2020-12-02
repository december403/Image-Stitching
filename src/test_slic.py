import numpy as np
from cv2 import cv2
import time

start_time = time.time()
img = cv2.imread('./image/UAV/DJI_0001.JPG')
# print(img)
img = cv2.GaussianBlur(img,(5,5),0)
img = cv2.cvtColor(img,cv2.COLOR_BGR2LAB)
superpixel = cv2.ximgproc.createSuperpixelSLIC(img,algorithm=cv2.ximgproc.SLIC, region_size=50)
superpixel.iterate(10)
superpixel.enforceLabelConnectivity(25)
number = superpixel.getNumberOfSuperpixels()
mask = superpixel.getLabelContourMask()
label_mask = superpixel.getLabels()

# img = cv2.cvtColor(img,cv2.COLOR_LAB2BGR)
# for idx in range(number):


#     img[label_mask==idx] = (np.random.randint(0,256),np.random.randint(0,256),np.random.randint(0,256))
# cv2.imwrite('slic.jpg', img)

print(label_mask)
print(number)
print(np.any(label_mask==7724))
print(f'-------{time.time()-start_time:8.3f} sec------------')