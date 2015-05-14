#!/usr/bin/python

import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import csv
import scipy.stats as stats

class pose_estimator():

    def __init__(self, cam_matrix, dist_matrix):

        self.dist_matrix = dist_matrix
        self.cam_matrix = cam_matrix

    def estimate_pose(self):

        search_size = (5, 4)

        axis = np.float32([[3, 0, 0], [0, 3, 0], [0, 0, 3]]).reshape(-1, 3)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        objp = np.zeros((search_size[0] * search_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:search_size[0], 0:search_size[1]].T.reshape(-1, 2)

        cap = cv2.VideoCapture('../videos/right_sd_test2.avi')

        trans = []
        rot = []

        n = 2400

        for i in range(n): 

            ret, frame = cap.read()
            grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            ret, corners = cv2.findChessboardCorners(grey, search_size, None)

            if ret:
                cv2.cornerSubPix(grey, corners, (11, 11), (-1, -1), criteria)
                rvecs, tvecs, inliers = cv2.solvePnPRansac(objp, corners, self.cam_matrix, self.dist_matrix)

                rot.append(self.find_euler_angles(rvecs))
                trans.append(self.find_pos(tvecs))
            else:
                print 'Corners not found in this frame'
                rot.append([0, 0, 0])
                trans.append([0, 0, 0])

            #cv2.imshow('frame', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        # Condition zero data points
        for i in range(1, n-1):
            if rot[i] == [0, 0, 0] and rot[i + 1] != [0, 0, 0]:
                rot[i] = [(rot[i-1][0] + rot[i + 1][0]) / 2, (rot[i-1][1] + rot[i + 1][1]) / 2, (rot[i-1][2] + rot[i + 1][2]) / 2]
                trans[i] = [(trans[i-1][0] + trans[i + 1][0]) / 2, (trans[i-1][1] + trans[i + 1][1]) / 2, (trans[i-1][2] + trans[i + 1][2]) / 2]

            elif rot[i] == [0, 0, 0] and rot[i + 1] == [0, 0, 0]:
                for j in range(i, n-1):
                    if rot[j] != [0, 0, 0]:
                        last_index = j
                        break
                if 'last_index' in locals():
                    for j in range(i, last_index):
                        rot[j] = [rot[j - 1][0] + ((rot[last_index][0] - rot[i - 1][0]) / (last_index - i + 1)),
                                  rot[j - 1][1] + ((rot[last_index][1] - rot[i - 1][1]) / (last_index - i + 1)),
                                  rot[j - 1][2] + ((rot[last_index][2] - rot[i - 1][2]) / (last_index - i + 1))]
                        trans[j] = [trans[j - 1][0] + ((trans[last_index][0] - trans[i - 1][0]) / (last_index - i + 1)),
                                    trans[j - 1][1] + ((trans[last_index][1] - trans[i - 1][1]) / (last_index - i + 1)),
                                    trans[j - 1][2] + ((trans[last_index][2] - trans[i - 1][2]) / (last_index - i + 1))]

        trans = np.asarray(trans)
        rot = np.asarray(rot)

        trans[:, 0] = [trans[:, 0][i]*-10  + 0 for i in range(0, len(trans[:, 0]))]
        trans[:, 1] = [trans[:, 1][i]*-10  + 0 for i in range(0, len(trans[:, 1]))]
        trans[:, 2] = [trans[:, 2][i]*-1  + 0 for i in range(0, len(trans[:, 2]))]

        rot[:, 0] = [rot[:, 0][i] - 0 for i in range(0, len(rot[:, 0]))]
        rot[:, 1] = [rot[:, 1][i]*-1 for i in range(0, len(rot[:, 1]))]
        rot[:, 2] = [rot[:, 2][i] - 90 if rot[:, 2][i] > 0 else rot[:, 2][i] + 90 for i in range(0, len(rot[:, 2]))]
        rot[:, 2] = [rot[:, 2][i] - 0 for i in range(0, len(rot[:, 2]))]

        temp_rot = np.zeros(rot.shape)
        temp_trans = np.zeros(trans.shape)

        # Make cam coordinate system coincide with Vicon coordinate system
        temp_trans[:, 0] = trans[:, 0]
        temp_trans[:, 1] = trans[:, 2]#*1.5
        temp_trans[:, 2] = trans[:, 1]

        temp_rot[:, 0] = rot[:, 2]
        temp_rot[:, 1] = rot[:, 1]
        temp_rot[:, 2] = rot[:, 0]

        return  temp_trans.T, temp_rot.T

    def find_pos(self, tvecs):
        return [15 * tvecs[0, 0], 15 * tvecs[1, 0], 15 * tvecs[2, 0]]

    def find_euler_angles(self, rvecs):
        rmat = cv2.Rodrigues(rvecs)[0]

        yaw = math.atan(rmat[1, 0] / rmat[0, 0])
        pitch = math.atan(-rmat[2, 0] / math.sqrt(rmat[2, 1] ** 2 + rmat[2, 2] ** 2))
        roll = math.atan(rmat[2, 1] / rmat[2, 2])

        return [math.degrees(roll), math.degrees(pitch), math.degrees(yaw)]

    def import_vicon_data(self, file_name):
        x = []
        y = []
        z = []
        roll = []
        pitch = []
        yaw = []

        index = 0
        x_t = 0
        y_t = 0
        z_t = 0
        roll_t = 0
        pitch_t = 0
        yaw_t = 0

        i = 0

        csvfile = csv.reader(open(file_name, 'r'))

        for row in csvfile:
            if not math.isnan(float(row[0])):
                x_t += float(row[0])
            if not math.isnan(float(row[1])):
                y_t += float(row[1])
            if not math.isnan(float(row[2])):
                z_t += float(row[2])
            if not math.isnan(float(row[3])):
                roll_t += float(row[3])
            if not math.isnan(float(row[4])):
                pitch_t += float(row[4])
            if not math.isnan(float(row[5])):
                yaw_t += float(row[5])

            index += 1

            if index % 10 == 0:
                x.append(x_t / 10.0)
                y.append(y_t / 10.0)
                z.append(z_t / 10.0)
                roll.append(roll_t / 10.0)
                pitch.append(pitch_t / 10.0)
                yaw.append(yaw_t / 10.0)

                x_t = 0
                y_t = 0
                z_t = 0
                roll_t = 0
                pitch_t = 0
                yaw_t = 0

        return_array = np.array([x, y, z, roll, pitch, yaw])

        return return_array[:, i:2400 + i]

    def save_to_csv(self, data):

        csv_writer = csv.writer(open('err_data.csv', 'w'))

        for i in range(6):
           csv_writer.writerow(data[i, :])

    def find_offset(self, trans, rot, vicon_data_board):

        e_x = trans[0, :] - vicon_data_board[0, :]
        e_y = trans[1, :] - vicon_data_board[1, :]
        e_z = trans[2, :] - vicon_data_board[2, :]
        e_roll = rot[0, :] - vicon_data_board[3, :]
        e_pitch = rot[1, :] - vicon_data_board[4, :]
        e_yaw = rot[2, :] - vicon_data_board[5, :]

        # Remove nan values
        e_x = e_x[~np.isnan(e_x)]
        e_y = e_y[~np.isnan(e_y)]
        e_z = e_z[~np.isnan(e_z)]
        e_roll = e_roll[~np.isnan(e_roll)]
        e_pitch = e_pitch[~np.isnan(e_pitch)]
        e_yaw = e_yaw[~np.isnan(e_yaw)]

        return np.array([[np.mean(e_x), np.mean(e_y), np.mean(e_z), np.mean(e_roll), np.mean(e_pitch), np.mean(e_yaw)]]).T

    def plot(self, data, tit):

        freq, bins = np.histogram(data, bins = 100)

        std = np.std(data)
        dist_range = np.arange(bins[0], bins[-1], 0.1)
        dist = stats.norm.pdf(dist_range, 0, std)

        plt.plot(dist_range, dist / float(max(dist)))

        width = 0.7 * (bins[1] - bins[0])
        center = (bins[:-1] + bins[1:]) / 2

        print freq
        print std
        print freq < std

        plt.bar(center, freq / float(max(freq[len(freq) / 4:len(freq) / 4 * 3])), align='center', width=width)
        plt.title(tit)

        plt.show()

    def plot_hist(self, data):

        self.plot(data[0, :], 'x')
        self.plot(data[1, :], 'y')
        self.plot(data[2, :], 'z')
        self.plot(data[3, :], 'roll')
        self.plot(data[4, :], 'pitch')
        self.plot(data[5, :], 'yaw')

    def is_normal(self, data):

        chi, p = stats.normaltest(data[0, :])
        print chi, p
        chi, p = stats.normaltest(data[1, :])
        print chi, p
        chi, p = stats.normaltest(data[2, :])
        print chi, p
        chi, p = stats.normaltest(data[3, :])
        print chi, p
        chi, p = stats.normaltest(data[4, :])
        print chi, p
        chi, p = stats.normaltest(data[5, :])
        print chi, p

    def cov(self, data):

        #print np.sqrt(np.fabs(np.cov(data)))
        print np.cov(data)

    def main(self):

         # Read training Vicon data
        vicon_data_board = self.import_vicon_data('../data/vicon_data/vicon_data_board.csv')
        vicon_data_cam = self.import_vicon_data('../data/vicon_data/vicon_data_cam.csv')
        vicon_data = vicon_data_board - vicon_data_cam

        trans, rot = self.estimate_pose()
        p_off = self.find_offset(trans, rot, vicon_data)

        err = np.concatenate((trans, rot)) - vicon_data - p_off

        self.plot_hist(err)
        #self.save_to_csv(err)
        #self.is_normal(err)
        #self.cov(err)

if __name__ == '__main__':

    with np.load('../calib_params/right_cam_calib_params.npz') as X:
                _, cam_matrix, distortion_matrix, r_vec, _ = [X[i] for i in ('ret', 'cam_matrix', 'distortion_matrix', 'r_vec', 't_vec')]

    cam_matrix[0, 0] = 780.0
    cam_matrix[1, 1] = 680.0
    pe = pose_estimator(cam_matrix, distortion_matrix)
    pe.main()
