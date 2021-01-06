import time
from cv2 import cv2
import numpy as np
from ImgMatcher import ImgMatcher


ref_img = cv2.imread('/home/zer0/image_stitching/image/level4/result00.png')
tar_img = cv2.imread('/home/zer0/image_stitching/image/level4/result02.png')
print(ref_img.shape)
print(tar_img.shape)


IM = ImgMatcher(tar_img,ref_img)
start_time = time.time()
IM.detectAKAZE()
print(f'finished detect in {time.time()-start_time}')
start_time = time.time()

IM.KNNmatchAKAZE()
print(f'finished matching in {time.time()-start_time}')
start_time = time.time()

    
with open('./data/matching_pairs/ORB_matching_pair.npy', 'wb') as f:
    np.save(f, IM.src_pts)
    np.save(f, IM.dst_pts)
 

print(f'There are {len(IM.tar_kps)} feature points in target image.')
print(f'There are {len(IM.ref_kps)} feature points in reference image.')
print(f'There are {len(IM.matches)} pairs matching pairs.')
print(f'There are {len(IM.mask[IM.mask == 1])} pairs matching pairs after RANSAC.')
# print(f"Process finished --- {(time.time() - start_time)} seconds ---")
img4 = img4 = cv2.drawMatches(tar_img, IM.tar_kps, ref_img, IM.ref_kps, IM.matches, None,
                       flags=cv2.DrawMatchesFlags_DEFAULT, matchesMask=IM.mask)
cv2.imwrite('Match.jpg', img4)
