
from cv2 import cv2
import numpy as np
from mask import Mask
import time
import matplotlib.pyplot as plt

def calculate_weight_map(tar_img, ref_img, mask):
    tar_img_YUV = cv2.GaussianBlur( cv2.cvtColor(tar_img,cv2.COLOR_BGR2YUV), (5,5), 10)
    ref_img_YUV = cv2.GaussianBlur( cv2.cvtColor(ref_img,cv2.COLOR_BGR2YUV), (5,5), 10)
    tar_img_Y = tar_img_YUV[:,:,0]
    ref_img_Y = ref_img_YUV[:,:,0]

    YUV_diff = np.abs( tar_img_YUV - ref_img_YUV)
    color_diff = cv2.convertScaleAbs( YUV_diff[:,:,0] * 0.5 + YUV_diff[:,:,1] * 0.25 + YUV_diff[:,:,2] * 0.25 )
    color_diff[mask.overlap==0] = 0

    print('finish YUV_diff')
    

    grad_diff_mag = getGradDiff(tar_img_Y,ref_img_Y)
    grad_diff_mag[mask.overlap==0] = 0
    print('finish grad_diff')


    color_grad_diff_sum = grad_diff_mag*100 + color_diff

    filter_bank = getGarborFilterBank(tar_img_Y, ref_img_Y)
    # print('finish gabor filter bank')

    h, w = tar_img_Y.shape
    tar_result = np.zeros((h,w))
    for i in range(len(filter_bank)):
        temp = cv2.filter2D(tar_img_Y, cv2.CV_64FC1, filter_bank[i])
        tar_result += temp**2 
    tar_result = np.sqrt(tar_result)
    tar_result[mask.overlap==0] = 0
    print('finish gabor feature target')

    ref_result = np.zeros((h,w))
    for i in range(len(filter_bank)):
        temp = cv2.filter2D(ref_img_Y, cv2.CV_64FC1, filter_bank[i])
        ref_result += temp**2

    ref_result = np.sqrt(ref_result)
    ref_result[mask.overlap==0] = 0
    print('finish gabor feature reference')

    gabor_result = ref_result + tar_result

    weight_map = np.multiply(gabor_result,  color_grad_diff_sum)
    print('finish weight map')

    # cv2.imwrite('./YUV_diff.png',color_diff)
    # cv2.imwrite('./gradian_diff.png',(grad_diff_mag/np.max(grad_diff_mag)*255).astype(np.uint8))
    # cv2.imwrite('./color_grad_diff.png',(color_grad_diff_sum/np.max(color_grad_diff_sum)*255).astype(np.uint8))
    # cv2.imwrite(f'./gabor/ref_gobar_final.png',(ref_result/np.max(ref_result)*255).astype(np.uint8) )
    # cv2.imwrite(f'./gabor/gobar_final.png',(gabor_result/np.max(gabor_result)*255).astype(np.uint8) )
    # cv2.imwrite(f'./gabor/tar_gobar_final.png',(tar_result/np.max(tar_result)*255).astype(np.uint8) )
    # cv2.imwrite(f'./gabor/W_final.png',(weight_map-np.mean(weight_map))/np.std(weight_map)*255)
    return weight_map

def getGradDiff(img1_Y, img2_Y):
    gradx = cv2.Sobel(img1_Y,cv2.CV_64F,1,0)
    grady = cv2.Sobel(img1_Y,cv2.CV_64F,0,1)

    grad_diff_mag = np.sqrt(gradx**2 + grady**2)
    gradx = cv2.Sobel(img2_Y,cv2.CV_64F,1,0)
    grady = cv2.Sobel(img2_Y,cv2.CV_64F,0,1)
    grad_diff_mag += np.sqrt(gradx**2 + grady**2)

    return grad_diff_mag


def getGarborFilterBank(tar_img_Y, ref_img_Y, ksize=127):
    rotate_angles = np.arange(0, np.pi, np.pi / 16)
    scales = np.array( [i*5 for i in range(1,2)] )

    gabor_filter_bank = []
    for i, angle in enumerate(rotate_angles):
        for j, scale in enumerate(scales):
            # gabor_filter_bank.append( cv2.getGaborKernel((5,5), 20, angle, scale, 10, ktype=cv2.CV_32F))
            # temp = cv2.getGaborKernel((ksize,ksize), 8, angle, scale, 0.5, psi=0, ktype=cv2.CV_32F)
            temp = cv2.getGaborKernel((ksize,ksize), scale*0.8, angle, scale, 0.5, psi=0, ktype=cv2.CV_32F)
            # temp = temp / np.sum(temp)
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

# ksize=127
# X = np.arange(0,ksize)
# Y = np.arange(0,ksize)
# X, Y = np.meshgrid(X, Y)
# rotate_angles = np.arange(0, np.pi, np.pi / 8)
# # scales = np.array( [np.exp(i*np.pi) for i in range(0,5)] )
# scales = np.array( [i*5 for i in range(1,6)] )
# # scales = np.array([5,10,15])
# for i, angle in enumerate(rotate_angles):
#     for j, scale in enumerate(scales):
#         # if i == 7 and j == 0:
#         if True:
#             # gabor_filter_bank.append( cv2.getGaborKernel((5,5), 20, angle, scale, 10, ktype=cv2.CV_32F))
#             # a = cv2.getGaborKernel((31,31), 5, angle, scale, 0.5, psi=0, ktype=cv2.CV_32F)
#             a = cv2.getGaborKernel((ksize,ksize), scale*0.8, angle, scale, 0.5, psi=0, ktype=cv2.CV_32F)
#             # print(f'filter{i}_{j} sum: {np.sum(a)}')
#             # a = a / np.sum(a)

#             # print(f'filter{i}_{j} max: {np.max(a)}')
#             # print(f'filter{i}_{j} min: {np.min(a)}')
#             print(f'filter{i}_{j} sum: {np.sum(a)}')
#             print()
#             fig = plt.figure()
#             axis = fig.gca(projection='3d')
#             axis.axes.set_zlim3d(bottom=-1,top=1)
#             # axis.set_proj_type('ortho')
#             surface = axis.plot_surface(X, Y, a, rstride=1, cstride=1, cmap='viridis')
#             # surface = axis.contour(X, Y, a, rstride=1, cstride=1,cmap='viridis')
#             # fig.colorbar(surface, shrink=1.0, aspect=20)
#             # plt.imshow( surface )
#             # fig.show()

#             fig.savefig(f'./gabor/0a{i}_{j}.png',dpi=300)
#             plt.clf()
#             # plt.show(block=True)
# # cv2.waitKey()

