#!/usr/bin/python

import numpy as np
import cv2
import math
import sys
import csv
import matplotlib.pyplot as plt

def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())

    cv2.line(img, corner, tuple(imgpts[0].ravel()), (255, 0, 0), 5)
    cv2.line(img, corner, tuple(imgpts[1].ravel()), (0, 255, 0), 5)
    cv2.line(img, corner, tuple(imgpts[2].ravel()), (0, 0, 255), 5)

    return img

def find_pos(tvecs):
    return [15 * tvecs[0, 0], 15 * tvecs[1, 0], 15 * tvecs[2, 0]]

def find_euler_angles(rvecs):
    rmat = cv2.Rodrigues(rvecs)[0]
    print rmat

    yaw = math.atan(rmat[1, 0] / rmat[0, 0])
    pitch = math.atan(-rmat[2, 0] / math.sqrt(rmat[2, 1] ** 2 + rmat[2, 2] ** 2))
    roll = math.atan(rmat[2, 1] / rmat[2, 2])

    return [math.degrees(roll), math.degrees(pitch), math.degrees(yaw)]

def main(argv):

    search_size = (5, 4)
    infile = 'videos/right_sd_test2.avi'
    outfile = 'data.csv'
    calib_file = 'right_cam_calib_params.npz'

    with np.load('calib_params/' + calib_file) as X:
        _, cam_matrix, distortion_matrix, r_vec, _ = [X[i] for i in ('ret', 'cam_matrix', 'distortion_matrix', 'r_vec', 't_vec')]

    axis = np.float32([[3, 0, 0], [0, 3, 0], [0, 0, 3]]).reshape(-1, 3)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((search_size[0] * search_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:search_size[0], 0:search_size[1]].T.reshape(-1, 2)

    objpoints = []
    imgpoints = []

    v_x = []
    v_y = []
    v_z = []
    v_roll = []
    v_pitch = []
    v_yaw = []

    index = 0
    v_x_t = 0
    v_y_t = 0
    v_z_t = 0
    v_roll_t = 0
    v_pitch_t = 0
    v_yaw_t = 0

    csvfile = csv.reader(open('vicon_data.csv', 'r'))
    for row in csvfile:
        v_x_t = v_x_t + float(row[0])
        v_y_t = v_y_t + float(row[1])
        v_z_t = v_z_t + float(row[2])
        v_roll_t = v_roll_t + float(row[3])
        v_pitch_t = v_pitch_t + float(row[4])
        v_yaw_t = v_yaw_t + float(row[5])

        index += 1

        if index == 10:
            v_x.append(v_x_t / 10.0)
            v_y.append(v_y_t / 10.0)
            v_z.append(v_z_t / 10.0)
            v_roll.append(v_roll_t / 10.0)
            v_pitch.append(v_pitch_t / 10.0)
            v_yaw.append(v_yaw_t / 10.0)

            index = 0
            v_x_t = 0
            v_y_t = 0
            v_z_t = 0
            v_roll_t = 0
            v_pitch_t = 0
            v_yaw_t = 0

    p_off = 2
    tran_diff = []
    rot_diff = []

    cap = cv2.VideoCapture(infile)
    t = range(1, 60)
    for i in range(t[-1]):
        ret, frame = cap.read()
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(grey, search_size, None)

        if ret:
            cv2.cornerSubPix(grey, corners, (11, 11), (-1, -1), criteria)
            rvecs, tvecs, inliers = cv2.solvePnPRansac(objp, corners, cam_matrix, distortion_matrix)
            imgpts, jacobian = cv2.projectPoints(axis, rvecs, tvecs, cam_matrix, distortion_matrix)

            #csvwriter.writerow(find_euler_angles(rvecs) + find_pos(tvecs))
            #print [v_roll[i], v_pitch[i], v_yaw[i]]
            print find_euler_angles(rvecs)[1], v_pitch[i]-90
            #print find_pos(tvecs)[2]*-2, v_y[i]
            rot_diff.append([find_euler_angles(rvecs)[2]+90 - v_roll[i], find_euler_angles(rvecs)[1] - v_pitch[i]+90, find_euler_angles(rvecs)[0] - v_yaw[i]])
            tran_diff.append([find_pos(tvecs)[0]*-10 - v_x[i], find_pos(tvecs)[2]*-2 - v_y[i], find_pos(tvecs)[1] - v_z[i]])

            frame = draw(frame, corners, imgpts)
        else:
            print 'Corners not found in this frame'
            #csvwriter.writerow((0, 0, 0, 0, 0, 0))

        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    x, = plt.plot(t, np.asarray(tran_diff)[:, 0])
    y, = plt.plot(t, np.asarray(tran_diff)[:, 1])
    z, = plt.plot(t, np.asarray(tran_diff)[:, 2])
    plt.legend([x, y, z], ['x', 'y', 'z'])
    plt.show()

    roll, = plt.plot(t, np.asarray(rot_diff)[:, 0])
    pitch, = plt.plot(t, np.asarray(rot_diff)[:, 1])
    yaw, = plt.plot(t, np.asarray(rot_diff)[:, 2])
    plt.legend([roll, pitch, yaw], ['roll', 'pitch', 'yaw'])
    plt.show()

if __name__ == '__main__':
    main(sys.argv[1:])