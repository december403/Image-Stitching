import pickle
from vertex import Vertex
from cv2 import cv2 
import numpy as np
from mask import Mask
from SLIC import MaskedSLIC
import time 

start = time.time()



with open('./data/SLIC/maskedSLIC.pkl', 'rb') as input_data:
    maskedSLIC = pickle.load(input_data)
tar_img = cv2.imread('./data/processed_image/warped_target.png')
ref_img = cv2.imread('./data/processed_image/warped_reference.png')
mask = Mask(tar_img, ref_img)
print(maskedSLIC.numOfPixel)
numOfPixel = maskedSLIC.numOfPixel
vertex = Vertex(0,maskedSLIC,mask)

# tar_img[vertex.y_coordi, vertex.x_coordi] = (0,0,0)

# print(f'The superpixel on target edge: {vertex.is_on_tar_edge}')
# print(f'The superpixel on reference edge: {vertex.is_on_ref_edge}')
# for label in range(numOfPixel):
for label in vertex.adjacent_vertices:
    adj_vertex = Vertex(label, maskedSLIC, mask)
    tar_img[adj_vertex.y_coordi, adj_vertex.x_coordi] = np.random.randint(0,255,size=(3,))
    # if adj_vertex.is_on_tar_edge:
    #     tar_img[adj_vertex.y_coordi, adj_vertex.x_coordi] = np.random.randint(0,255,size=(3,))


print(time.time() - start)
cv2.imwrite('aaaaa.png', tar_img)