import pickle
from vertex import Vertex
from cv2 import cv2 
import numpy as np
from mask import Mask
from SLIC import MaskedSLIC
import time 
from edge_weight import calculate_weight_map, getGarborFilterBank
import maxflow

start = time.time()



with open('./data/SLIC/maskedSLIC.pkl', 'rb') as input_data:
    maskedSLIC = pickle.load(input_data)
# for i in maskedSLIC.adjacent_pairs:
#     print(i)
tar_img = cv2.imread('./data/processed_image/warped_target.png')
ref_img = cv2.imread('./data/processed_image/warped_reference.png')
mask = Mask(tar_img, ref_img)
weight_map = calculate_weight_map(tar_img, ref_img, mask)
# print(maskedSLIC.numOfPixel)
numOfPixel = maskedSLIC.numOfPixel
verteices_lst = []
for idx in range(1,numOfPixel):
    vertex = Vertex(idx,maskedSLIC,mask,weight_map)
    # print(vertex.weight)
    verteices_lst.append(vertex)

graph = maxflow.Graph[float](numOfPixel-1, 3*numOfPixel)
nodes = graph.add_nodes(numOfPixel-1)
print(nodes)
for idx, vertex in enumerate(verteices_lst):
        for adj_vert_idx in vertex.adjacent_vertices:
            edge_weight = vertex.weight + verteices_lst[adj_vert_idx-1].weight
            graph.add_edge(idx, adj_vert_idx-1, edge_weight, edge_weight)

for idx, vertex in enumerate(verteices_lst):
    source_w = 0
    sink_w = 0
    if vertex.is_on_ref_edge:
        source_w = 20000000000
    if vertex.is_on_tar_edge:
        sink_w = 20000000000
    graph.add_tedge(idx,source_w, sink_w)
print(graph.maxflow())

result = tar_img + ref_img
mask_seam = np.zeros(result.shape[0:2])
for idx, vertex,in enumerate(verteices_lst):
    if  graph.get_segment(idx) :
        result[vertex.y_coordi, vertex.x_coordi] = tar_img[vertex.y_coordi, vertex.x_coordi]
    else:
        mask_seam[vertex.y_coordi, vertex.x_coordi] = 255
        result[vertex.y_coordi, vertex.x_coordi] = ref_img[vertex.y_coordi, vertex.x_coordi]
cv2.imwrite('aaaaa.png', result)
cv2.imwrite('aseam.png', mask_seam)
# print(graph.)


# tar_img[vertex.y_coordi, vertex.x_coordi] = (0,0,0)

# print(f'The superpixel on target edge: {vertex.is_on_tar_edge}')
# print(f'The superpixel on reference edge: {vertex.is_on_ref_edge}')
# for label in range(numOfPixel):
# for label in vertex.adjacent_vertices:
#     adj_vertex = Vertex(label, maskedSLIC, mask)
#     tar_img[adj_vertex.y_coordi, adj_vertex.x_coordi] = np.random.randint(0,255,size=(3,))
    # if adj_vertex.is_on_tar_edge:
    #     tar_img[adj_vertex.y_coordi, adj_vertex.x_coordi] = np.random.randint(0,255,size=(3,))


# print(time.time() - start)
# cv2.imwrite('aaaaa.png', tar_img)