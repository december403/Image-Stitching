import numpy as np
from cv2 import cv2
from matplotlib import pyplot as plt
import itertools as it
import cupy as cp
import sys


def findLocalHomography_SVD(src_pts, dst_pts, grids_coordi):
    '''
    This function calculate the location dependent homography matrix for each grids.
    src_pts : All target image's paired key points' x y coordinate. it's a Nx2 matrix, N is number of matching pairs
    dst_pts : All reference image's paired key points' x y coordinate. it's Nx2 matrix, N is number of matching pairs
    '''
    grid_num = len(grids_coordi)
    # W_list = [ get_W(src_pts, grid_coordi) for grid_coordi in grids_coordi]

    src_mean = np.mean(src_pts, axis=0)
    src_std = max(np.std(src_pts, axis=0))
    C1 = np.array([[1/src_std,         0, -src_mean[0]/src_std],
                   [0, 1/src_std, -src_mean[1]/src_std],
                   [0,         0,                    1]])
    src_pts = np.hstack((src_pts, np.ones((len(src_pts), 1))))
    src_pts = src_pts @ C1.T

    dst_mean = np.mean(dst_pts, axis=0)
    dst_std = max(np.std(dst_pts, axis=0))
    C2 = np.array([[1/dst_std,         0, -dst_mean[0]/dst_std],
                   [0, 1/dst_std, -dst_mean[1]/dst_std],
                   [0,         0,                    1]])
    dst_pts = np.hstack((dst_pts, np.ones((len(dst_pts), 1))))
    dst_pts = dst_pts @ C2.T

    A = np.zeros((2*len(src_pts), 9))
    for i in range(len(src_pts)):
        x1, y1, _ = src_pts[i]
        x2, y2, _ = dst_pts[i]
        A[i*2, :] = [0, 0, 0, -x1, -y1, -1, y2*x1, y2*y1, y2]
        A[i*2+1, :] = [x1, y1, 1, 0, 0, 0, -x2*x1, -x2*y1, -x2]

    local_Hs = np.zeros((grid_num, 3, 3))

    # for idx, W in enumerate(W_list):
    for idx in range(grid_num):

        print(f'SVD {idx:8d} / {grid_num}   {idx/grid_num*100:8.1f}%', end='\r')
        W = cp.asarray(get_W(src_pts, grids_coordi[idx]))
        A = cp.asarray(A)
        u, s, v = cp.linalg.svd(W @ A)
        H = v[-1, :].reshape((3, 3))
        H = cp.asnumpy(H)
        H = np.linalg.inv(C2) @ H @ C1
        H = H/H[-1, -1]
        local_Hs[idx, :, :] = H
    return local_Hs


def findLocalHomography_SVD_CUDA(src_pts, dst_pts, grids_coordi):
    '''
    This function calculate the location dependent homography matrix for each grids.
    src_pts : All target image's paired key points' x y coordinate. it's a Nx2 matrix, N is number of matching pairs
    dst_pts : All reference image's paired key points' x y coordinate. it's Nx2 matrix, N is number of matching pairs
    '''
    print(grids_coordi.shape)

    src_pts_unnormalized = np.copy(src_pts)
    src_pts_unnormalized = cp.asarray(src_pts_unnormalized)
    grids_coordi = cp.asarray(grids_coordi)
    grid_num = len(grids_coordi)
    src_mean = np.mean(src_pts, axis=0)
    src_std = max(np.std(src_pts, axis=0))
    C1 = np.array([[1/src_std,         0, -src_mean[0]/src_std],
                   [0, 1/src_std, -src_mean[1]/src_std],
                   [0,         0,                    1]])
    src_pts = np.hstack((src_pts, np.ones((len(src_pts), 1))))
    src_pts = src_pts @ C1.T

    dst_mean = np.mean(dst_pts, axis=0)
    dst_std = max(np.std(dst_pts, axis=0))
    C2 = np.array([[1/dst_std,         0, -dst_mean[0]/dst_std],
                   [0, 1/dst_std, -dst_mean[1]/dst_std],
                   [0,         0,                    1]])
    dst_pts = np.hstack((dst_pts, np.ones((len(dst_pts), 1))))
    dst_pts = dst_pts @ C2.T

    A = np.zeros((2*len(src_pts), 9))
    for i in range(len(src_pts)):
        x1, y1, _ = src_pts[i]
        x2, y2, _ = dst_pts[i]
        A[i*2, :] = [0, 0, 0, -x1, -y1, -1, y2*x1, y2*y1, y2]
        A[i*2+1, :] = [x1, y1, 1, 0, 0, 0, -x2*x1, -x2*y1, -x2]

    local_Hs = np.zeros((grid_num, 3, 3))
    A = cp.asarray(A)
    scale_factor = 20
    src_pts = cp.asarray(src_pts)
    matchNum = src_pts.shape[0]

    u, s, v = cp.linalg.svd(A)
    H = v[-1, :].reshape((3, 3))
    H = cp.asnumpy(H)
    H = np.linalg.inv(C2) @ H @ C1
    H = H/H[-1, -1]
    H_unweighted = H[...]
    skip = 0
    unskip_grids = []
    for idx in range(grid_num):
        print(f'SVD {idx:8d}/{grid_num}({idx/grid_num*100:8.1f}%)  Current skip {skip} times. Current Skip rate is {skip/grid_num:5.3%}', end='\r')
        grid_coordi = grids_coordi[idx]

        weight = cp.exp(
            (-1) * cp.sum((src_pts_unnormalized - grid_coordi)**2, axis=1) / scale_factor**2)

        if cp.amax(weight) < 0.025:
            skip += 1
            local_Hs[idx, :, :] = H_unweighted
            continue
        unskip_grids.append(grid_coordi)
        weight = cp.repeat(weight, 2)
        weight[weight < 0.025] = 0.025
        weight = weight.reshape((2*matchNum, 1))
        weighted_A = cp.multiply(weight, A)
        u, s, v = cp.linalg.svd(weighted_A)
        H = v[-1, :].reshape((3, 3))
        H = cp.asnumpy(H)
        H = np.linalg.inv(C2) @ H @ C1
        H = H/H[-1, -1]
        local_Hs[idx, :, :] = H
    print()
    print(f'Skip {skip} times. Skip rate is {skip/grid_num:5.3%}')

    return local_Hs, unskip_grids


def get_W(src_pts, grid_coordi):
    scale_factor = 15
    # scale_factor = 5000
    # scale_factor =np.amax( np.linalg.norm( src_pts[:,0:2] - grid_coordi, axis=1 ) )
    weight = np.exp(
        (-1) * np.linalg.norm(src_pts[:, 0:2] - grid_coordi, axis=1) / scale_factor**2)

    # weight[ weight < 0.025] = 0.025
    # print(f'The max weight is {np.amax(weight)}')
    # print(f'The min weight is {np.amin(weight)}')
    # print(f'weight 0~20: {weight[0:20]}')
    # print()
    # # print(scale_factor)
    # print('max ', np.amax(  np.linalg.norm( src_pts[:,0:2] - grid_coordi, axis=1 ))  )
    # print()
    W = np.diag(np.repeat(weight, 2))
    return W


def perspectiveTransform_local(local_homo_matrix, grid_topleft_coordi, grid_size):
    x = np.repeat(np.arange(grid_size).reshape(1, grid_size), grid_size,
                  axis=0).reshape((grid_size**2)) + grid_topleft_coordi[0]
    y = np.repeat(np.arange(grid_size).reshape(grid_size, 1), grid_size,
                  axis=1).reshape((grid_size**2)) + grid_topleft_coordi[1]
    ones = np.ones((grid_size**2))
    src_coordi = np.row_stack((x, y, ones))
    # print(local_homo_matrix.shape)
    # print(src_coordi.shape)

    dst_coordi = local_homo_matrix @ src_coordi
    dst_coordi = dst_coordi / dst_coordi[2, :]

    return src_coordi[0:2, :].astype(np.int16), dst_coordi[0:2, :].astype(np.int16)


def findMatchPair_ORB(tar_img, ref_img, ptsNum=10000, save=False, fileName='ORB_matching_pair.npy', RANSAC=True, projError=50):
    mask = None
    H = None
    # Initiate ORB detector

    orb = cv2.ORB_create(ptsNum)
    # find the keypoints and descriptors with ORB
    ref_kp, ref_des = orb.detectAndCompute(ref_img, None)
    tar_kp, tar_des = orb.detectAndCompute(tar_img, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(ref_des, tar_des)
    src_pts = np.array([tar_kp[match.trainIdx].pt for match in matches])
    dst_pts = np.array([ref_kp[match.queryIdx].pt for match in matches])
    if RANSAC:
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, projError)
        src_pts = np.array([tar_kp[match.trainIdx].pt for idx,
                            match in enumerate(matches) if mask[idx] == 1])
        dst_pts = np.array([ref_kp[match.queryIdx].pt for idx,
                            match in enumerate(matches) if mask[idx] == 1])


    return src_pts, dst_pts, tar_kp, ref_kp, matches, mask


def findMatchPair_GMS(tar_img, ref_img, ptsNum=10000, save=False, fileName='GMS_matching_pair.npy', RANSAC=True, projError=50):
    # Initiate ORB detector
    orb = cv2.ORB_create(ptsNum, fastThreshold=0)
    # find the keypoints and descriptors with ORB
    ref_kp, ref_des = orb.detectAndCompute(ref_img, None)
    tar_kp, tar_des = orb.detectAndCompute(tar_img, None)

    # bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.match(ref_des, tar_des)
    matches_GMS = cv2.xfeatures2d.matchGMS(
        ref_img.shape[0:2], tar_img.shape[0:2], ref_kp, tar_kp, matches, withRotation=True)
    src_pts = np.array([tar_kp[match.trainIdx].pt for match in matches_GMS])
    dst_pts = np.array([ref_kp[match.queryIdx].pt for match in matches_GMS])
    if RANSAC:
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, projError)
        src_pts = np.array([tar_kp[match.trainIdx].pt for idx,
                            match in enumerate(matches_GMS) if mask[idx] == 1])
        dst_pts = np.array([ref_kp[match.queryIdx].pt for idx,
                            match in enumerate(matches_GMS) if mask[idx] == 1])

    if save:
        with open(fileName, 'wb') as f:
            np.save(f, src_pts)
            np.save(f, dst_pts)

        print('GMS matching pairs saved')
    return src_pts, dst_pts, tar_kp, ref_kp, matches_GMS, mask


# class Grid
