from cv2 import cv2 
import numpy as np
from  APAP_Sticher import APAP_Stitcher
import time
from mask import Mask


ref_img = cv2.imread('./DJI/DJI_0001.JPG')
tar_img = cv2.imread('./DJI/DJI_0002.JPG')

if ref_img is None:
    print(f'File ./DJI/DJI_0001.JPG not found')


with open('GMS_matching_pair.npy', 'rb') as f:
    src_pts = np.load(f)
    dst_pts = np.load(f)


stitcher = APAP_Stitcher(tar_img, ref_img, src_pts, dst_pts, grid_size=100, scale_factor=15)

stitcher.homoMat.constructGlobalMat(stitcher.src_pts, stitcher.dst_pts)
stitcher.homoMat.constructLocalMat(src_pts, stitcher.grids, 15)
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


start_time = time.time()

# cv2.warpPerspective(tar_img, shift@H, dsize=(x,y), dst=warp_tar_img, borderMode=cv2.BORDER_TRANSPARENT)
cv2.warpPerspective(ref_img, shift, dsize=(x,y), dst=warp_ref_img, borderMode=cv2.BORDER_TRANSPARENT)
mask = Mask(warp_tar_img,warp_tar_img,warp_ref_img)

grid_num = stitcher.grids.number
# for idx in stitcher.homoMat.non_global_homo_mat_lst:
for idx, local_H in enumerate(stitcher.homoMat.localHomoMat_lst):
    x1, y1 = stitcher.grids.topLeft_lst[idx]
    x2, y2 = stitcher.grids.botRight_lst[idx]
    # local_H = stitcher.homoMat.localHomoMat_lst[idx]
    shift2 = np.zeros((3,3))
    shift2[0,2] = x1
    shift2[1,2] = y1
    shift2[2,2] = 1
    shift2[0,0] = 1
    shift2[1,1] = 1
    cv2.warpPerspective(tar_img[y1:y2+2, x1:x2+2,:], shift@local_H@shift2, dsize=(x,y), dst=warp_tar_img, borderMode=cv2.BORDER_TRANSPARENT)
    print(f'Warped grids {idx+1:8d}/{grid_num}({(idx+1)/(grid_num)*100:8.1f}%)', end='\r')



print(time.time() - start_time)
result = np.zeros((y,x,3),np.uint8)
# result[mask.overlap>0] = cv2.addWeighted(warp_tar_img,0.5,warp_ref_img,0.5,0)[mask.overlap>0]
result = warp_tar_img

# result[mask.ref_nonoverlap >0] = warp_ref_img[mask.ref_nonoverlap>0]
# result[mask.tar_nonoverlap >0] = warp_tar_img[mask.tar_nonoverlap>0]

print(result.shape)
cv2.imwrite('aa.jpg',result)
# cv2.imwrite('aa.jpg',mask.overlap)


