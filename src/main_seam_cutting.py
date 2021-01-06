import pickle
from vertex import Vertex
from cv2 import cv2 
import numpy as np
from mask import Mask
from SLIC import MaskedSLIC
import time 
from edge_weight import calculate_weight_map, getGarborFilterBank
import maxflow
import argparse





parser = argparse.ArgumentParser()
parser.add_argument('out_file_name')
args = parser.parse_args()




start = time.time()

tar_img = cv2.imread('./warped_target.png')
ref_img = cv2.imread('./warped_reference.png')
mask = Mask(tar_img, ref_img)
cv2.imwrite('overlap_mask.png', mask.overlap)

weight_map = calculate_weight_map(tar_img, ref_img, mask)

cv2.imwrite('weight_map.png', weight_map/np.max(weight_map)*255)

print(f'finished weight map calculation   in {time.time()-start:8.3f}')
start = time.time()
# exit(0)
maskedSLIC = MaskedSLIC(ref_img, mask.overlap, region_size=10)
print(f'finished SLIC calculation   in {time.time()-start:8.3f}')
start = time.time()

numOfPixel = maskedSLIC.numOfPixel
verteices_lst = []
for idx in range(1,numOfPixel):
    vertex = Vertex(idx,maskedSLIC,mask,weight_map)
    verteices_lst.append(vertex)

print(f'finished verteices construction   in {time.time()-start:8.3f}')
start = time.time() 

graph = maxflow.Graph[float](numOfPixel-1, 3*numOfPixel)
nodes = graph.add_nodes(numOfPixel-1)



for vertex_i, vertex_j in maskedSLIC.adjacent_pairs:
    edge_weight = verteices_lst[vertex_i-1].weight + verteices_lst[vertex_j-1].weight
    graph.add_edge(vertex_i-1, vertex_j-1, edge_weight, edge_weight)

print(f'finished n link construction   in {time.time()-start:8.3f}')
start = time.time() 


for idx, vertex in enumerate(verteices_lst):
    source_w = 0
    sink_w = 0
    if vertex.is_on_ref_edge:
        source_w = 20000000000
    if vertex.is_on_tar_edge:
        sink_w = 20000000000
    graph.add_tedge(idx,source_w, sink_w)

print(f'finished graph construction   in {time.time()-start:8.3f}')
start = time.time()
graph.maxflow()
print(f'finished graph cut   in {time.time()-start:8.3f}')
start = time.time()

result = tar_img + ref_img
mask_seam = np.zeros(result.shape[0:2])
for idx, vertex,in enumerate(verteices_lst):
    if  graph.get_segment(idx) :
        result[vertex.y_coordi, vertex.x_coordi] = tar_img[vertex.y_coordi, vertex.x_coordi]
    else:
        mask_seam[vertex.y_coordi, vertex.x_coordi] = 255
        result[vertex.y_coordi, vertex.x_coordi] = ref_img[vertex.y_coordi, vertex.x_coordi]





cv2.imwrite(args.out_file_name, result)
result[mask.overlap_edge>0] = (0,255,0)
# result[maskedSLIC.contour_mask>0] = (0,255,0)
# cv2.imwrite('result_pixel.png', result)
kernal = np.ones((5,5), np.int8)
mask_seam = cv2.morphologyEx(mask_seam, cv2.MORPH_DILATE, kernal) - mask_seam
mask_seam[mask.overlap==0] = 0

result[mask_seam>0] = (0,0,255)

cv2.imwrite('result_pixel_seam.png', result)

print(f'finished stitching   in {time.time()-start:8.3f}')

