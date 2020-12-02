from func import *
import time
start_time = time.time()
ref_img = cv2.imread('./DJI/DJI_0001.JPG')
tar_img = cv2.imread('./DJI/DJI_0002.JPG')

src_pts, dst_pts, kp2, kp1, matches, mask = findMatchPair_ORB(
    tar_img, ref_img, save=True, ptsNum=200000, projError=5)

# src_pts, dst_pts, kp2, kp1, matches, mask = findMatchPair_ORB(
#     tar_img, ref_img, save=True, ptsNum=200000, projError=50)
    
img4 = cv2.drawMatches(ref_img, kp1, tar_img, kp2, matches, None,
                       flags=cv2.DrawMatchesFlags_DEFAULT, matchesMask=mask)
with open('GMS_matching_pair.npy', 'wb') as f:
    np.save(f, src_pts)
    np.save(f, dst_pts)
 


print(f'There are {len(matches)} pairs matching pairs.')
print(f'There are {len(mask[mask == 1])} pairs matching pairs after RANSAC.')
# cv2.imwrite('ORB_Match_RANSAC.jpg', img4)
cv2.imwrite('GMS_Match_RANSAC.jpg', img4)
print(f"Process finished --- {(time.time() - start_time)} seconds ---")
