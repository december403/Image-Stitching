from cv2 import cv2 
import numpy as np
from  APAP_Sticher import APAP_Stitcher
import time
from mask import Mask
from ImgMatcher import ImgMatcher
import argparse

# parser = argparse.ArgumentParser()
# parser.add_argument('target_image')
# parser.add_argument('reference_image')

# args = parser.parse_args()

tar_img = cv2.imread('image/level0/DJI_0016_dark.png')
ref_img = cv2.imread('image/level0/DJI_0015.png')

# ref_img = cv2.imread('image/paper_test_image/1.jpg')
# tar_img = cv2.imread('image/paper_test_image/2.jpg')
# ref_img = cv2.imread(args.reference_image)
# tar_img = cv2.imread(args.target_image)
# ref_img = cv2.imread('/home/zer0/image_stitching/2.jpg')
# tar_img = cv2.imread('/home/zer0/image_stitching/1.jpg')

IM = ImgMatcher(tar_img,ref_img)
IM.detectAKAZE()
IM.KNNmatchAKAZE()

src_pts = IM.src_pts
dst_pts = IM.dst_pts
print('finish matching')
# exit(0)
stitcher = APAP_Stitcher(tar_img, ref_img, src_pts, dst_pts, grid_size=30, scale_factor=15)
stitcher.homoMat.constructGlobalMat(stitcher.src_pts, stitcher.dst_pts)
print('finish H')

# stitcher.homoMat.constructLocalMat(src_pts, stitcher.grids, 15)
stitched_img_size, shift_amount = stitcher.find_stitched_img_size_and_shift_amount(tar_img, ref_img)
x, y = stitched_img_size
H = stitcher.homoMat.globalHomoMat
shift = np.zeros((3,3))
shift[0,2] = shift_amount[0]
shift[1,2] = shift_amount[1]
shift[2,2] = 1
shift[0,0] = 1
shift[1,1] = 1


warp_tar_img = np.zeros((y,x,3),np.uint8)
warp_ref_img = np.zeros((y,x,3),np.uint8)

# print(f'Calculate homography matrices time: {time.time() - start_time:8.5f}')

# start_time = time.time()

cv2.warpPerspective(tar_img, shift@H, dsize=(x,y), dst=warp_tar_img, borderMode=cv2.BORDER_TRANSPARENT)
cv2.warpPerspective(ref_img, shift, dsize=(x,y), dst=warp_ref_img, borderMode=cv2.BORDER_TRANSPARENT)
mask = Mask(warp_tar_img,warp_ref_img)

# grid_num = stitcher.grids.number
# for idx in stitcher.homoMat.non_global_homo_mat_lst:
# # for idx, local_H in enumerate(stitcher.homoMat.localHomoMat_lst):
#     x1, y1 = stitcher.grids.topLeft_lst[idx]
#     x2, y2 = stitcher.grids.botRight_lst[idx]
#     local_H = stitcher.homoMat.localHomoMat_lst[idx]
#     shift2 = np.zeros((3,3))
#     shift2[0,2] = x1
#     shift2[1,2] = y1
#     shift2[2,2] = 1
#     shift2[0,0] = 1
#     shift2[1,1] = 1
#     cv2.warpPerspective(tar_img[y1:y2+2, x1:x2+2,:], shift@local_H@shift2, dsize=(x,y), dst=warp_tar_img, borderMode=cv2.BORDER_TRANSPARENT)
#     print(f'Warpping grids number {idx+1:6d}/{grid_num}', end='\r')

# print()

# print(f'warping time: {time.time() - start_time:8.5f}')
result = np.zeros((y,x,3),np.uint8)
result[mask.overlap>0] = cv2.addWeighted(warp_tar_img,0.5,warp_ref_img,0.5,0)[mask.overlap>0]
result[mask.ref_nonoverlap >0] = warp_ref_img[mask.ref_nonoverlap>0]
result[mask.tar_nonoverlap >0] = warp_tar_img[mask.tar_nonoverlap>0]

# new_h, new_w = warp_ref_img.shape[0:2]
# new_h = new_h//8
# new_w = new_w//8
# warp_ref_img = cv2.resize(warp_ref_img,(new_w,new_h))
# warp_tar_img = cv2.resize(warp_tar_img,(new_w,new_h))
cv2.imwrite('./warped_reference.png',warp_ref_img)
cv2.imwrite('./warped_target.png',warp_tar_img)
# cv2.imwrite('./simple_average_result.png',result)
np.save('save_H.npy',H)
np.save('save_Shift.npy',shift)
np.save('save_ref_4_corners_xy.npy', np.array( [ [0,0], [0,ref_img.shape[0]], [ref_img.shape[1],0], [ref_img.shape[1],ref_img.shape[0]] ] ) )
np.save('save_tar_4_corners_xy.npy', np.array( [ [0,0], [0,tar_img.shape[0]], [tar_img.shape[1],0], [tar_img.shape[1],tar_img.shape[0]] ] ) )
