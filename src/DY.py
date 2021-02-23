def constructGlobalMat(self, src_pts, dst_pts):
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


        u, s, v = np.linalg.svd(A)
        H = v[-1, :].reshape((3, 3))
        H = np.linalg.inv(C2) @ H @ C1
        H = H/H[-1, -1]
        return H
        # self.globalHomoMat = H
        # self.A = A
        # self.C1 = C1
        # self.C2 = C2

def find_stitched_img_size_and_shift_amount(self, tar_img, ref_img, H):
        '''
        This method finds the height and width of final stitched image.
        '''
        if self.homoMat.globalHomoMat is None:
            raise ValueError(' Run constructGlobalMat() before stitch the image!')

        

        tar_h, tar_w = tar_img.shape[0:2]
        tar_four_corners = np.array( [ (0,0), (0,tar_h), (tar_w,0), (tar_w,tar_h)] )

        ref_h, ref_w = ref_img.shape[0:2]
        ref_four_coeners = np.array( [ (0,0), (0,ref_h), (ref_w,0), (ref_w,ref_h)] )

        # H = self.homoMat.globalHomoMat
        warped_tar_four_corners = np.zeros((4,2))
        for idx, (x, y) in enumerate(tar_four_corners):
            temp = H @ np.array([x,y,1]).reshape((3,1))
            temp = temp/temp[2,0]
            warped_tar_four_corners[idx] = temp[0:2,0]

        eight_corners = np.vstack((warped_tar_four_corners, ref_four_coeners))

        min_x, min_y = np.min(eight_corners, axis=0).astype(np.int16)
        max_x, max_y = np.max(eight_corners, axis=0).astype(np.int16)
        shift_amount = np.array( (-min_x, -min_y)).reshape((2,1)).astype(np.int16)
        stitched_img_size = np.array((int(max_x-min_x+1), int(max_y-min_y+1)))
        return stitched_img_size, shift_amount