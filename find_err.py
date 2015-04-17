#!/usr/bin/python

import numpy as np
import math
import cv2
import csv
import matplotlib.pyplot as plt

def estimate_pose(cam_matrix, distortion_matrix, T):

    search_size = (5, 4)

    axis = np.float32([[3, 0, 0], [0, 3, 0], [0, 0, 3]]).reshape(-1, 3)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((search_size[0] * search_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:search_size[0], 0:search_size[1]].T.reshape(-1, 2)

    cap = cv2.VideoCapture('videos/test.avi')

    trans = []
    rot = []

    for i in range(T):
        ret, frame = cap.read()
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(grey, search_size, None)

        if ret:
            cv2.cornerSubPix(grey, corners, (11, 11), (-1, -1), criteria)
            rvecs, tvecs, inliers = cv2.solvePnPRansac(objp, corners, cam_matrix, distortion_matrix)
            imgpts, jacobian = cv2.projectPoints(axis, rvecs, tvecs, cam_matrix, distortion_matrix)

            frame = draw(frame, corners, imgpts)

            rot.append(find_euler_angles(rvecs))
            trans.append(find_pos(tvecs))
        else:
            print 'Corners not found in this frame'
            rot.append([0, 0, 0])
            trans.append([0, 0, 0])

        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    #plt.plot(np.asarray(rot)[:, 0])

    # Condition zero data points
    for i in range(1, T-1):
        if rot[i] == [0, 0, 0] and rot[i + 1] != [0, 0, 0]:
            rot[i] = [(rot[i-1][0] + rot[i + 1][0]) / 2, (rot[i-1][1] + rot[i + 1][1]) / 2, (rot[i-1][2] + rot[i + 1][2]) / 2]
            trans[i] = [(trans[i-1][0] + trans[i + 1][0]) / 2, (trans[i-1][1] + trans[i + 1][1]) / 2, (trans[i-1][2] + trans[i + 1][2]) / 2]

        elif rot[i] == [0, 0, 0] and rot[i + 1] == [0, 0, 0]:
            for j in range(i, T-1):
                if rot[j] != [0, 0, 0]:
                    last_index = j
                    break
            for j in range(i, last_index):
                rot[j] = [rot[j - 1][0] + ((rot[last_index][0] - rot[i - 1][0]) / (last_index - i + 1)),
                          rot[j - 1][1] + ((rot[last_index][1] - rot[i - 1][1]) / (last_index - i + 1)),
                          rot[j - 1][2] + ((rot[last_index][2] - rot[i - 1][2]) / (last_index - i + 1))]
                trans[j] = [trans[j - 1][0] + ((trans[last_index][0] - trans[i - 1][0]) / (last_index - i + 1)),
                            trans[j - 1][1] + ((trans[last_index][1] - trans[i - 1][1]) / (last_index - i + 1)),
                            trans[j - 1][2] + ((trans[last_index][2] - trans[i - 1][2]) / (last_index - i + 1))]

    # plt.plot(np.asarray(rot)[:, 0])
    # plt.show()

    return  np.asarray(trans), np.asarray(rot)

def find_offset(trans, rot, vicon_data, T):
    # v_x = []
    # v_y = []
    # v_z = []
    # v_roll = []
    # v_pitch = []
    # v_yaw = []
    #
    # index = 0
    # v_x_t = 0
    # v_y_t = 0
    # v_z_t = 0
    # v_roll_t = 0
    # v_pitch_t = 0
    # v_yaw_t = 0
    #
    # csvfile = csv.reader(open('vicon_data.csv', 'r'))
    #
    # for row in csvfile:
    #     v_x_t = v_x_t + float(row[0])
    #     v_y_t = v_y_t + float(row[1])
    #     v_z_t = v_z_t + float(row[2])
    #     v_roll_t = v_roll_t + float(row[3])
    #     v_pitch_t = v_pitch_t + float(row[4])
    #     v_yaw_t = v_yaw_t + float(row[5])
    #
    #     index += 1
    #
    #     if index == 10:
    #         v_x.append(v_x_t / 10.0)
    #         v_y.append(v_y_t / 10.0)
    #         v_z.append(v_z_t / 10.0)
    #         v_roll.append(v_roll_t / 10.0)
    #         v_pitch.append(v_pitch_t / 10.0)
    #         v_yaw.append(v_yaw_t / 10.0)
    #
    #         index = 0
    #         v_x_t = 0
    #         v_y_t = 0
    #         v_z_t = 0
    #         v_roll_t = 0
    #         v_pitch_t = 0
    #         v_yaw_t = 0

    v_x = vicon_data[0, :]
    #print v_x
    v_y = vicon_data[1, :]
    v_z = vicon_data[2, :]
    v_roll = vicon_data[3, :]
    v_pitch = vicon_data[4, :]
    v_yaw = vicon_data[5, :]

    trans[:, 0] = [trans[:, 0][i]*-10 for i in range(0, len(trans[:, 0]))]
    trans[:, 1] = [trans[:, 1][i]*-10 + 550 for i in range(0, len(trans[:, 1]))]
    trans[:, 2] = [trans[:, 2][i]*-10 + 2650 for i in range(0, len(trans[:, 2]))]

    rot[:, 1] = [rot[:, 1][i]*-1 + 88 for i in range(0, len(rot[:, 1]))]
    rot[:, 2] = [rot[:, 2][i] - 90 if rot[:, 2][i] > 0 else rot[:, 2][i] + 90 for i in range(0, len(rot[:, 2])) ]

    e_x = trans[:, 0] - v_x
    e_y = trans[:, 2] - v_y
    e_z = trans[:, 1] - v_z
    e_roll = rot[:, 2] - v_roll
    e_pitch = rot[:, 1] - v_pitch
    e_yaw = rot[:, 0] - v_yaw#np.asarray(v_yaw[300:300+T])

    # Remove nan values
    # e_x = e_x[~np.isnan(e_x)]
    # e_y = e_y[~np.isnan(e_y)]
    # e_z = e_z[~np.isnan(e_z)]
    # e_roll = e_roll[~np.isnan(e_roll)]
    # e_pitch = e_pitch[~np.isnan(e_pitch)]
    # e_yaw = e_yaw[~np.isnan(e_yaw)]

    #print trans[:, 0]
    #print trans[:, 0].shape, v_x.shape
    plt.plot(trans[:, 1], 'g')
    plt.plot(v_z, 'r')
    plt.show()

    # plt.plot(e_x, 'r')
    # plt.plot(e_y, 'g')
    # plt.plot(e_z, 'b')
    # plt.plot(e_roll, 'k')
    # plt.plot(e_pitch, 'c')
    # plt.plot(e_yaw, 'm')
    # plt.show()

    return [np.mean(e_x), np.mean(e_y), np.mean(e_z), np.mean(e_roll), np.mean(e_pitch), np.mean(e_yaw)]

def import_vicon_data(T):
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

        if index % 10 == 0:
            # Condition NaN numbers
            if math.isnan(v_x_t):
                v_x_t = (v_x[index%10 - 1] + v_x[index%10 - 2]) / 2.0

            if math.isnan(v_y_t):
                v_y_t = (v_y[index%10 - 1] + v_y[index%10 - 2]) / 2.0

            if math.isnan(v_z_t):
                v_z_t = (v_z[index%10 - 1] + v_z[index%10 - 2]) / 2.0

            if math.isnan(v_roll_t):
                v_roll_t = (v_roll[index%10 - 1] + v_roll[index%10 - 2]) / 2.0

            if math.isnan(v_pitch_t):
                v_pitch_t = (v_pitch[index%10 - 1] + v_pitch[index%10 - 2]) / 2.0

            if math.isnan(v_yaw_t):
                v_yaw_t = (v_yaw[index%10 - 1] + v_yaw[index%10 - 2]) / 2.0

            v_x.append(v_x_t / 10.0)
            v_y.append(v_y_t / 10.0)
            v_z.append(v_z_t / 10.0)
            v_roll.append(v_roll_t / 10.0)
            v_pitch.append(v_pitch_t / 10.0)
            v_yaw.append(v_yaw_t / 10.0)

            v_x_t = 0
            v_y_t = 0
            v_z_t = 0
            v_roll_t = 0
            v_pitch_t = 0
            v_yaw_t = 0

    return_array = np.array([v_x, v_y, v_z, v_roll, v_pitch, v_yaw])

    return return_array[:, 300  :300+T]

def find_err(vicon_data, p_off, cam_matrix, distortion_matrix, T):

    # v_x = []
    # v_y = []
    # v_z = []
    # v_roll = []
    # v_pitch = []
    # v_yaw = []
    #
    # index = 0
    # v_x_t = 0
    # v_y_t = 0
    # v_z_t = 0
    # v_roll_t = 0
    # v_pitch_t = 0
    # v_yaw_t = 0
    #
    # csvfile = csv.reader(open('vicon_data.csv', 'r'))
    #
    # for row in csvfile:
    #     v_x_t = v_x_t + float(row[0])
    #     v_y_t = v_y_t + float(row[1])
    #     v_z_t = v_z_t + float(row[2])
    #     v_roll_t = v_roll_t + float(row[3])
    #     v_pitch_t = v_pitch_t + float(row[4])
    #     v_yaw_t = v_yaw_t + float(row[5])
    #
    #     index += 1
    #
    #     if index == 10:
    #         v_x.append(v_x_t / 10.0)
    #         v_y.append(v_y_t / 10.0)
    #         v_z.append(v_z_t / 10.0)
    #         v_roll.append(v_roll_t / 10.0)
    #         v_pitch.append(v_pitch_t / 10.0)
    #         v_yaw.append(v_yaw_t / 10.0)
    #
    #         index = 0
    #         v_x_t = 0
    #         v_y_t = 0
    #         v_z_t = 0
    #         v_roll_t = 0
    #         v_pitch_t = 0
    #         v_yaw_t = 0


    f_x = range(int(math.ceil(cam_matrix[0, 0] / 10.0) * 10) - 50, int(math.ceil(cam_matrix[0, 0] / 10.0) * 10) + 60, 10)
    f_y = range(int(math.ceil(cam_matrix[1, 1] / 10.0) * 10) - 50, int(math.ceil(cam_matrix[1, 1] / 10.0) * 10) + 60, 10)
    min_err = 10000000
    for x in f_x:
        for y in f_y:
            print 'Working on focus ' + str(x) + '_' + str(y)
            cam_matrix[0, 0] = x
            cam_matrix[1, 1] = y

            trans, rot = estimate_pose(cam_matrix, distortion_matrix, T)
            trans[:, 0] = [trans[:, 0][i]*-10 + 0 for i in range(0, len(trans[:, 0]))]
            trans[:, 1] = [trans[:, 1][i]*-10 + 711 for i in range(0, len(trans[:, 1]))]
            trans[:, 2] = [trans[:, 2][i]*-10 + 2950 for i in range(0, len(trans[:, 2]))]

            rot[:, 1] = [rot[:, 1][i]*-1 + 90 for i in range(0, len(rot[:, 1]))]
            rot[:, 2] = [rot[:, 2][i]    - 90 if rot[:, 2][i] > 0 else rot[:, 2][i] + 90 for i in range(0, len(rot[:, 2]))]

            cam_data = np.concatenate((trans, rot), axis = 1).T

            #print cam_data
            #print np.isnan(vicon_data).any()
            #print p_off
            err = cam_data - vicon_data - p_off
            err_sum = np.sqrt(np.sum(err ** 2, axis = 1))
            #print err_sum.shape

            if np.linalg.norm(err_sum) < min_err:
                print np.linalg.norm(err_sum)
                print 'Min err at ' + str(x) + '_' + str(y)
                min_err = np.mean(err_sum)
                min_x = x
                min_y = y
        #     plt.plot(trans[:, 0], 'r')
        # plt.plot(vicon_data[0, :], 'g')
            plt.plot(err_sum)
    plt.show()
            #plt.legend([err_p], [str(x) + '_' + str(y)])

    #plt.show()

    return min_x, min_y

def main():
    # Find initial cam matrix

    with np.load('calib_params/right_cam_calib_params.npz') as X:
            _, cam_matrix, distortion_matrix, r_vec, _ = [X[i] for i in ('ret', 'cam_matrix', 'distortion_matrix', 'r_vec', 't_vec')]

    T = 100
    cam_mat_orig = cam_matrix

    # Read Vicon data
    vicon_data = import_vicon_data(T)

    for i in range(3):
        # Step 1: Find pose with focus length f
        trans, rot = estimate_pose(cam_matrix, distortion_matrix, T)

        # Step 2 + 3: Determine avg error
        p_off = find_offset(trans, rot, vicon_data, T)
        p_off = np.array([p_off]).T

        # Step 4: Minimise err by varying focus length f
        min_x, min_y = find_err(vicon_data, p_off, cam_matrix, distortion_matrix, T)

        #Step 5: Adapt cam_matrix with new focal length and repeat
        cam_matrix[0, 0] = min_x
        cam_matrix[1, 1] = min_y

    # Original
    trans_1, rot_1 = estimate_pose(cam_mat_orig, distortion_matrix, T)
    # draw_errors(rot, trans, vicon_data, T)

    # Improved
    cam_matrix_min_err = cam_matrix
    print min_x, min_y
    cam_matrix_min_err[0, 0] = min_x
    cam_matrix_min_err[1, 1] = min_y
    trans, rot = estimate_pose(cam_matrix_min_err, distortion_matrix, T)
    #draw_errors(rot, trans, vicon_data, T)

    # plt.plot(vicon_data[0, :])
    # plt.show()

    draw_comparrisson(rot, trans, rot_1, trans_1, vicon_data)


def draw_comparrisson(i_rot, i_trans, o_rot, o_trans, vicon_data):
    i_trans[:, 0] = [i_trans[:, 0][i]*-10 + 0 for i in range(0, len(i_trans[:, 0]))]
    i_trans[:, 1] = [i_trans[:, 1][i]*-10 + 711 for i in range(0, len(i_trans[:, 1]))]
    i_trans[:, 2] = [i_trans[:, 2][i]*-10 + 2950 for i in range(0, len(i_trans[:, 2]))]

    o_trans[:, 0] = [o_trans[:, 0][i]*-10 + 0 for i in range(0, len(o_trans[:, 0]))]
    o_trans[:, 1] = [o_trans[:, 1][i]*-10 + 711 for i in range(0, len(o_trans[:, 1]))]
    o_trans[:, 2] = [o_trans[:, 2][i]*-10 + 2950 for i in range(0, len(o_trans[:, 2]))]

    i_rot[:, 1] = [i_rot[:, 1][i]*-1 + 88 for i in range(0, len(i_rot[:, 1]))]
    i_rot[:, 2] = [i_rot[:, 2][i] - 90 if i_rot[:, 2][i] > 0 else i_rot[:, 2][i] + 90 for i in range(0, len(i_rot[:, 2])) ]

    o_rot[:, 1] = [o_rot[:, 1][i]*-1 + 88 for i in range(0, len(o_rot[:, 1]))]
    o_rot[:, 2] = [o_rot[:, 2][i] - 90 if o_rot[:, 2][i] > 0 else o_rot[:, 2][i] + 90 for i in range(0, len(o_rot[:, 2]))]

    titles = ['x', 'y', 'z', 'roll', 'pitch', 'yaw']

    o_p, = plt.plot(o_trans[:, 0], 'r:')
    i_p, = plt.plot(i_trans[:, 0], 'g--')
    v_p, = plt.plot(vicon_data[0, :], 'b_')
    plt.legend([o_p, i_p, v_p], ['Original', 'Improved', 'Vicon'])
    plt.title(titles[0])
    plt.show()

    o_p, = plt.plot(o_trans[:, 1], 'r:')
    i_p, = plt.plot(i_trans[:, 1], 'g--')
    v_p, = plt.plot(vicon_data[2, :], 'b_')
    plt.legend([o_p, i_p, v_p], ['Original', 'Improved', 'Vicon'])
    plt.title(titles[2])
    plt.show()

    o_p, = plt.plot(o_trans[:, 2], 'r:')
    i_p, = plt.plot(i_trans[:, 2], 'g--')
    v_p, = plt.plot(vicon_data[1, :], 'b_')
    plt.legend([o_p, i_p, v_p], ['Original', 'Improved', 'Vicon'])
    plt.title(titles[1])
    plt.show()

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

    # C = np.array([[-0.0303272, -0.0006465, -0.9995398],
    #               [-0.0447882, 0.9989963,   0.0007128],
    #               [0.9985361,  0.0447892,  -0.0303257]])
    # print C.shape, rmat.shape
    # rmat = rmat.dot(C)

    yaw = math.atan(rmat[1, 0] / rmat[0, 0])
    pitch = math.atan(-rmat[2, 0] / math.sqrt(rmat[2, 1] ** 2 + rmat[2, 2] ** 2))
    roll = math.atan(rmat[2, 1] / rmat[2, 2])

    return [math.degrees(roll), math.degrees(pitch), math.degrees(yaw)]

if __name__ == '__main__':
    main()