from cv2 import cv2
import numpy as np
from mask import Mask
import time
import matplotlib.pyplot as plt

def calculate_weight_map(tar_img, ref_img, mask):
    tar_img_YUV = cv2.GaussianBlur( cv2.cvtColor(tar_img,cv2.COLOR_BGR2YUV), (3,3), 0)
    ref_img_YUV = cv2.GaussianBlur( cv2.cvtColor(ref_img,cv2.COLOR_BGR2YUV), (3,3), 0)
    tar_img_Y = tar_img_YUV[:,:,0]
    ref_img_Y = ref_img_YUV[:,:,0]

    YUV_diff = np.abs( tar_img_YUV - ref_img_YUV)
    color_diff = cv2.convertScaleAbs( YUV_diff[:,:,0] * 0.9 + YUV_diff[:,:,1] * 0.05 + YUV_diff[:,:,2] * 0.05 )
    color_diff[mask.overlap==0] = 0
    cv2.imwrite('./YUV_diff.png',YUV_diff[:,:,0])


    Y_diff = YUV_diff[:,:,0]
    gradx = cv2.convertScaleAbs(cv2.Sobel(Y_diff,cv2.CV_64F,1,0))
    grady = cv2.convertScaleAbs(cv2.Sobel(Y_diff,cv2.CV_64F,0,1))
    grad_diff_mag = cv2.addWeighted(gradx, 0.5, grady, 0.5, 0)
    grad_diff_mag[mask.overlap==0] = 0
    # cv2.imwrite('./Y_gradian_diff.png',(grad_diff_mag).astype(np.uint8))

    color_grad_diff_sum = cv2.addWeighted(grad_diff_mag, 1, color_diff, 0.5, 0)
    # cv2.imwrite('./color_grad_diff.png',(color_grad_diff_sum.astype(np.uint8)) )

    filter_bank = getGarborFilterBank(tar_img_Y, ref_img_Y)
    h, w = tar_img_Y.shape
    tar_result = np.zeros((h,w))
    for i in range(len(filter_bank)):
        temp = cv2.filter2D(tar_img_Y, cv2.CV_8UC1, filter_bank[i])
        # tar_result = np.maximum(temp, tar_result)
        tar_result += temp
        # cv2.imwrite(f'./gabor/feature_map_{i}.jpeg',temp)#/np.max(temp)*255 )
    # tar_result[mask.overlap==0] = 0
    # cv2.imwrite(f'./gabor/tar_gobar_final.png',tar_result )
    
    ref_result = np.zeros((h,w))
    for i in range(len(filter_bank)):
        temp = cv2.filter2D(ref_img_Y, cv2.CV_8UC1, filter_bank[i])
        # ref_result = np.maximum(temp, ref_result)
        ref_result += temp
    # ref_result[mask.overlap==0] = 0
    # cv2.imwrite(f'./gabor/ref_gobar_final.png',ref_result )

    gabor_result = ref_result + tar_result
    # cv2.imwrite(f'./gabor/gobar_final.png',gabor_result )

    weight_map = np.multiply(gabor_result,  color_grad_diff_sum)
    return weight_map
    cv2.imwrite(f'./gabor/W_final.png',(weight_map-np.mean(weight_map))/np.std(weight_map)*255)





def getGarborFilterBank(tar_img_Y, ref_img_Y):
    rotate_angles = np.arange(0, np.pi, np.pi / 8)
    scales = np.array( [np.exp(np.pi*i) for i in range(0,5)] )
    # scales = np.array([0.1,0.5,1,5,10])
    # scales = np.array([1,3,5,7,30])

    gabor_filter_bank = []
    for i, angle in enumerate(rotate_angles):
        for j, scale in enumerate(scales):
            # gabor_filter_bank.append( cv2.getGaborKernel((5,5), 20, angle, scale, 10, ktype=cv2.CV_32F))
            temp = cv2.getGaborKernel((31,31), 8, angle, scale, 0.5, psi=0, ktype=cv2.CV_32F)
            temp = temp / (np.sum(temp)*1.5)
            gabor_filter_bank.append(temp)
            # plt.imshow(temp)
            # plt.savefig(f'./gabor/0gabor_filter_{i}_{j}.png')
            # print(i)
            # print(temp)
            # print(i)

    return gabor_filter_bank



# start = time.time()
# tar_img = cv2.imread('./data/processed_image/warped_target.png')
# ref_img = cv2.imread('./data/processed_image/warped_reference.png')
# mask = Mask(tar_img, ref_img)
# calculate_weight_map(tar_img, ref_img, mask)

# print(time.time() - start)



# rotate_angles = np.arange(0, np.pi, np.pi / 8)
# scales = np.array( [np.exp(i*np.pi) for i in range(0,5)] )
# for i, angle in enumerate(rotate_angles):
#     for j, scale in enumerate(scales):
#         # gabor_filter_bank.append( cv2.getGaborKernel((5,5), 20, angle, scale, 10, ktype=cv2.CV_32F))
#         a = cv2.getGaborKernel((31,31), 5, angle, 10, 0.5, psi=0, ktype=cv2.CV_32F)
#         a = a / np.sum(a) / 1.5
#         plt.imshow(cv2.getGaborKernel((127,127), 20, angle, 10, 0.5, psi=0, ktype=cv2.CV_32F))
#         plt.savefig(f'./gabor/a{i}_{j}.png')