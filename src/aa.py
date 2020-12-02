
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

# with open('ORB_matching_pair.npy', 'rb') as f:
with open('GMS_matching_pair.npy', 'rb') as f:
    src_pts = np.load(f)
    dst_pts = np.load(f)
 



for i in range(1, 20):
    if np.gcd(w, h) % i == 0:
        grid_size = i
print(f'the grid size is {grid_size}x{grid_size}')

# Draw grid on source image
tar_img_grid = np.copy(tar_img)
for row in np.arange(0, w, grid_size):
    cv2.line(tar_img_grid, (row, 0), (row, h), (0, 255, 0), 1)
for column in np.arange(0, h, grid_size):
    cv2.line(tar_img_grid, (0, column), (w, column), (0, 255, 0), 1)




# calculate grid's center coordinate and top grid's left coordinate
grids_center_coordi = it.product(np.arange(
    0, w, grid_size)+grid_size//2, np.arange(0, h, grid_size)+grid_size//2)
grids_center_coordi = np.array([x for x in grids_center_coordi])
grids_topleft_coordi = it.product(
    np.arange(0, w, grid_size), np.arange(0, h, grid_size))
grids_topleft_coordi = np.array([x for x in grids_topleft_coordi])
print(f'There are {len(grids_center_coordi)} grids.')
print(f'The number of matching pairs is {len(src_pts)}')

# for i in grids_center_coordi:
#     print(i)

# calculate grids' local dependent homography matrices
local_homo_matrices, unskip_grids = findLocalHomography_SVD_CUDA(
    src_pts, dst_pts, grids_center_coordi)
for grid_center in unskip_grids:
    grid_center = cp.asnumpy(grid_center)
    cv2.circle(tar_img_grid, tuple(grid_center), 5, (0,255,0), -1)

for x,y in src_pts:
    cv2.circle(tar_img_grid, (int(x),int(y)), 3, (0,0,255), -1)  
cv2.imwrite('highlight_unskip_grid.jpg', tar_img_grid)





max_x = np.NINF
max_y = np.NINF
min_x = np.Inf
min_y = np.Inf
src = []
dst = []
print(f'The number of homo matrix is {len(local_homo_matrices)}')

for local_H, grid_coordi in zip(local_homo_matrices, grids_topleft_coordi):
    src_coordi, dst_coordi = perspectiveTransform_local(
        local_H, grid_coordi, grid_size)

    src.append(src_coordi)
    dst.append(dst_coordi)
    if max_x < np.amax(dst_coordi[0, :]):
        max_x = np.amax(dst_coordi[0, :])
    if max_y < np.amax(dst_coordi[1, :]):
        max_y = np.amax(dst_coordi[1, :])
    if min_x > np.amin(dst_coordi[0, :]):
        min_x = np.amin(dst_coordi[0, :])
    if min_y > np.amin(dst_coordi[1, :]):
        min_y = np.amin(dst_coordi[1, :])

# print(f'Warped target image max x is {max_x}')
# print(f'Warped target image max y is {max_y}')
# print(f'Warped target image min x is {min_x}')
# print(f'Warped target image min y is {min_y}')
# print(f'Reference image max x is {ref_img.shape[1]}')
# print(f'Reference image max y is {ref_img.shape[0]}')
# print(f'Reference image min x is {0}')
# print(f'Reference image min y is {0}')
# print(f'Final panaroma max x is {max(max_x,ref_img.shape[1])}')
# print(f'Final panaroma max y is {max(max_y,ref_img.shape[0])}')
# print(f'Final panaroma min x is {min(min_x,0)}')
# print(f'Final panaroma min y is {min(min_y,0)}')
x_shift = -min(min_x, 0)
y_shift = -min(min_y, 0)
print('AAAAA', (  max(max_y, ref_img.shape[0])-min(min_y, 0)+1, max(max_x, ref_img.shape[1])-min(min_x, 0)+1, 3  ) )
result_grid = np.zeros(
    (  max(max_y, ref_img.shape[0])-min(min_y, 0)+1, max(max_x, ref_img.shape[1])-min(min_x, 0)+1, 3  )  )

result = np.zeros(
    (max(max_y, ref_img.shape[0])-min(min_y, 0)+1, max(max_x, ref_img.shape[1])-min(min_x, 0)+1, 3))

for src_coordi, dst_coordi in zip(src, dst):

    # result[dst_coordi[1, :]+y_shift, dst_coordi[0, :] +
    #        x_shift] = np.round( tar_img[src_coordi[1, :], src_coordi[0, :]] ,decimals=1)
    # result[dst_coordi[1, :]+y_shift, dst_coordi[0, :] +
    #        x_shift] = np.round( tar_img[src_coordi[1, :], src_coordi[0, :]], decimals=1)

    result_grid[dst_coordi[1, :]+y_shift, dst_coordi[0, :] +
           x_shift] = np.round( tar_img_grid[src_coordi[1, :], src_coordi[0, :]] ,decimals=1)
    result_grid[dst_coordi[1, :]+y_shift, dst_coordi[0, :] +
           x_shift] = np.round( tar_img_grid[src_coordi[1, :], src_coordi[0, :]], decimals=1)

    result[dst_coordi[1, :]+y_shift, dst_coordi[0, :] +
           x_shift] = np.round( tar_img[src_coordi[1, :], src_coordi[0, :]] ,decimals=1)
    result[dst_coordi[1, :]+y_shift, dst_coordi[0, :] +
           x_shift] = np.round( tar_img[src_coordi[1, :], src_coordi[0, :]], decimals=1)

    # result[dst_coordi[1,:]+y_shift, dst_coordi[0,:]+x_shift] = tar_img_grid[ src_coordi[1,:], src_coordi[0,:]]
    # result[dst_coordi[1,:]+y_shift, dst_coordi[0,:]+x_shift] = tar_img_grid[ src_coordi[1,:], src_coordi[0,:]]


cv2.imwrite('APAP_target_image.jpg', result.astype(np.int16))
cv2.imwrite('APAP_target_image_grid.jpg', result_grid.astype(np.int16))


ref_imgh, ref_imgw = ref_img.shape[0:2]
ref_img_shift = np.zeros(result.shape)
# print(result.shape)
# print(ref_img.shape)
ref_img_shift[y_shift:y_shift+ref_imgh, x_shift:x_shift+ref_imgw, :] = ref_img
# plt.imshow(ref_img_shift.astype(np.int16));plt.show()
cv2.imwrite('APAP_reference_image.jpg', ref_img_shift.astype(np.int16))

ref_img_mask = ref_img_shift > 0
result_mask = result > 0
overlap_mask = np.bitwise_and(result_mask, ref_img_mask)
# overlap_mask = result_mask *ref_img_mask
# print(overlap_mask.shape)
final = (ref_img_shift + result) // 2 * overlap_mask
final = (ref_img_shift + result) - final
# plt.imshow(final.astype(np.int16));plt.show()
cv2.imwrite('APAP_final.jpg', final.astype(np.int16))

print(f"Process finished --- {(time.time() - start_time)} seconds ---")


# convert -loop 0 -delay 10 APAP_reference_image.jpg APAP_target_image.jpg APAP_overlap.gif