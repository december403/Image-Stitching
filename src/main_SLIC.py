from cv2 import cv2 
import numpy as np
from mask import Mask
import time
from SLIC import MaskedSLIC
import pickle

start_time = time.time()

tar_img = cv2.imread('./data/processed_image/warped_target.png')
ref_img = cv2.imread('./data/processed_image/warped_reference.png')
mask = Mask(tar_img, ref_img)
maskedSLIC = MaskedSLIC(tar_img, mask.overlap, region_size=50)



for (rows, cols) in maskedSLIC.labels_position:
    tar_img[rows, cols] = np.random.randint(0,256,size=(1,3))

cv2.imwrite('./slic_pandas.png',tar_img)
print(time.time() - start_time)
print(f'number of pixel is :{maskedSLIC.numOfPixel}')

with open('./data/SLIC/maskedSLIC.pkl', 'wb') as output:
    pickle.dump(maskedSLIC, output, pickle.HIGHEST_PROTOCOL)

