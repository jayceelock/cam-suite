#!/usr/bin/python

import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import getopt

from subprocess import call

class ErrorFinder:

    def __init__(self, f_x, f_y):
        self.start = True
        self.f_x = f_x
        self.f_y = f_y

    def estimate_pose(self, cam_matrix, dist_mat):
        
        search_size = (5, 4)
        
        axis = np.float32(([[3, 0, 0], [0, 3, 0], [0, 0, 3]])).reshape(-1, 3)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        objp = np.zeros((search_size[0] * search_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:search_size[0], 0:search_size[1]].T.reshape(-1, 2)

        cap = cv2.VideoCapture('../videos/right_sd_test2.avi')

        while(cap.isOpened()):
            
            ret, frame = cap.read()
            grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            ret, corners = cv2.findChessboardCorners(grey, search_size, None)

            if ret:
                cv2.cornerSubPix(grey, corners, (11, 11), (-1, -1), criteria)
                rvecs, tvecs, inliers = cv2.solvePnPRansac(objp, corners, cam_matrix, dist_mat)

            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    def main(self):
        print 'hey'

        with np.load('../calib_params/right_cam_calib_params.npz') as X:
            _, cam_matrix, distortion_matrix, r_vec, _ = [X[i] for i in ('ret', 'cam_matrix', 'distortion_matrix', 'r_vec', 't_vec')]

        cam_matrix[0, 0] = self.f_x
        cam_matrix[1, 1] = self.f_y

        self.estimate_pose(cam_matrix, distortion_matrix)

def required_args_present(opts):

    required_args = ['-x', '-y']
    test = 0

    for opt, arg in opts:
        if opt in required_args:
            test += 1
    if test == 2:
        return True
    return False

if __name__ == '__main__':
    
    argv = sys.argv[1:]

    try:
        opts, args = getopt.getopt(argv, 'x:y:', ['fx=', 'fy'])
    except getopt.GetoptError:
        print 'Usage: find_error.py -fx <number> -fy <number>'
        sys.exit(2)
    if not required_args_present(opts):
        print '-x and -y paramaters required.'
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-x':
            f_x = float(arg)
        elif opt == '-y':
            f_y = float(arg)

    ef = ErrorFinder(f_x, f_y)
    ef.main()
