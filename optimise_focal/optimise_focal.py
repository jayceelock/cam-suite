#!/usr/bin/python

import numpy as np
import math
import cv2
import cv2.cv as cv
import csv
import matplotlib.pyplot as plt
import random

class ErrFinder():

    def __init__(self):
        self.camdata = None
        self.errdata = None
        self.vicondata = None
        
        self.tr_n = 90
        self.ts_n = 2400
        
        self.samples = random.sample(range(0, self.ts_n), self.tr_n)

        self.min_err = 10000000
        self.min_x = 700.0
        self.min_y = 700.0

    def estimate_pose(self, cam_matrix, distortion_matrix, n, training = True):

        search_size = (5, 4)

        axis = np.float32([[3, 0, 0], [0, 3, 0], [0, 0, 3]]).reshape(-1, 3)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        objp = np.zeros((search_size[0] * search_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:search_size[0], 0:search_size[1]].T.reshape(-1, 2)

        cap = cv2.VideoCapture('../videos/vicon_videos/right_sd_test2.avi')

        trans = []
        rot = []

        for i in range(n):

            if training:
                cap.set(cv.CV_CAP_PROP_POS_FRAMES, self.samples[i])
                ret, frame = cap.read()
            else:
                ret, frame = cap.read()
            #cv2.imshow('img', frame)
            grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            ret, corners = cv2.findChessboardCorners(grey, search_size, None)

            if ret:
                cv2.cornerSubPix(grey, corners, (11, 11), (-1, -1), criteria)
                if n == 0:  
                    rvecs, tvecs, inliers = cv2.solvePnPRansac(objp, corners, cam_matrix, distortion_matrix, rvec=rvecs, tvec=tvecs, useExtrinsicGuess=True)
                else:
                    rvecs, tvecs, inliers = cv2.solvePnPRansac(objp, corners, cam_matrix, distortion_matrix)
                # imgpts, jacobian = cv2.projectPoints(axis, rvecs, tvecs, cam_matrix, distortion_matrix)

                # frame = draw(frame, corners, imgpts)

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
        trans[:, 2] = [trans[:, 2][i]*-10  + 0 for i in range(0, len(trans[:, 2]))]

        rot[:, 0] = [rot[:, 0][i] - 0 for i in range(0, len(rot[:, 0]))]
        rot[:, 1] = [rot[:, 1][i]*-1 for i in range(0, len(rot[:, 1]))]
        rot[:, 2] = [rot[:, 2][i] - 90 if rot[:, 2][i] > 0 else rot[:, 2][i] + 90 for i in range(0, len(rot[:, 2]))]
        rot[:, 2] = [rot[:, 2][i] - 0 for i in range(0, len(rot[:, 2]))]

        temp_rot = np.zeros(rot.shape)
        temp_trans = np.zeros(trans.shape)

        # Make cam coordinate system coincide with Vicon coordinate system
        temp_trans[:, 0] = trans[:, 0]
        temp_trans[:, 1] = trans[:, 2]
        temp_trans[:, 2] = trans[:, 1]

        temp_rot[:, 0] = rot[:, 2]
        temp_rot[:, 1] = rot[:, 1]
        temp_rot[:, 2] = rot[:, 0]

        return  temp_trans.T, temp_rot.T

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

    def import_vicon_data(self, file_name, n):
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

        return return_array[:, i:n + i]

    def find_err(self, vicon_data, p_off, cam_matrix, distortion_matrix, n):

        # Intervals of 10
        #f_x = range(int(math.ceil(cam_matrix[0, 0] / 10.0) * 10) - 50, int(math.ceil(cam_matrix[0, 0] / 10.0) * 10) + 60, 10)
        #f_y = range(int(math.ceil(cam_matrix[1, 1] / 10.0) * 10) - 50, int(math.ceil(cam_matrix[1, 1] / 10.0) * 10) + 60, 10)

        f_x = range(int(cam_matrix[0, 0]) - 5, int(cam_matrix[0, 0]) + 5, 1)
        f_y = range(int(cam_matrix[1, 1]) - 5, int(cam_matrix[1, 1]) + 5, 1)

        min_err_sum = 0

        for x in f_x:
            for y in f_y:
                print 'Working on focus ' + str(x) + '_' + str(y)
                cam_matrix[0, 0] = x
                cam_matrix[1, 1] = y

                trans, rot = self.estimate_pose(cam_matrix, distortion_matrix, self.tr_n)

                cam_data = np.concatenate((trans, rot), axis = 0)

                err = cam_data - vicon_data - p_off

                if self.camdata == None:
                    self.camdata = cam_data
                    self.vicondata = vicon_data
                    self.errdata = err

                else:
                    self.camdata = np.concatenate((self.camdata, cam_data), axis = 1)
                    self.vicondata = np.concatenate((self.vicondata, vicon_data), axis = 1)
                    self.errdata = np.concatenate((self.errdata, err), axis = 1)

                # Scale the errors by dividing by their expected maxima
                err[0, :] = err[0, :] / 1000.0
                err[1, :] = err[1, :] / 3000.0
                err[2, :] = err[2, :] / 1000.0

                err_sum = np.sqrt(np.sum(err[:3, :] ** 2, axis = 1))

                print 'Error:' + str(err_sum)
                print 'Norm:' + str(np.linalg.norm(err_sum))

                if np.linalg.norm(err_sum) < self.min_err:
                    print 'Min err at ' + str(x) + '_' + str(y)
                    self.min_err = np.linalg.norm(err_sum)
                    min_err_sum = np.linalg.norm(err_sum)
                    self.min_x = x
                    self.min_y = y

        return self.min_x, self.min_y, min_err_sum

    def main(self):
        # Find initial cam matrix

        with np.load('../calib_params/right_cam_calib_params.npz') as X:
                _, cam_matrix, distortion_matrix, r_vec, _ = [X[i] for i in ('ret', 'cam_matrix', 'distortion_matrix', 'r_vec', 't_vec')]


        # Original
        trans_1, rot_1 = self.estimate_pose(cam_matrix, distortion_matrix, self.ts_n, training = False)

        # Read training Vicon data
        vicon_data_board = self.import_vicon_data('../data/vicon_data/vicon_data_board.csv', self.ts_n)
        vicon_data_cam = self.import_vicon_data('../data/vicon_data/vicon_data_cam.csv', self.ts_n)
        vicon_data = vicon_data_board - vicon_data_cam

        # Select tr_n random data points from the Vicon data set and ensure they are normally distributed
        tr_vicon_data = np.asarray([[vicon_data[i, j] for j in self.samples] for i in range(6)])

        e_t = []

        for i in range(20):
            # Step 1: Find pose with focus length f
            trans, rot = self.estimate_pose(cam_matrix, distortion_matrix, self.tr_n)

            # Step 2 + 3: Determine avg error
            p_off = self.find_offset(trans, rot, tr_vicon_data)
            print 'p_off:' + str(p_off)
            cam = np.concatenate((trans, rot), axis = 0)
            off_err = cam - tr_vicon_data - p_off
            #e_t.append(off_err)

            # Step 4: Minimise err by varying focus length f
            min_x, min_y, focus_err = self.find_err(tr_vicon_data, p_off, cam_matrix, distortion_matrix, self.tr_n)
            #e_t.append(focus_err)

            #Step 5: Adapt cam_matrix with new focal length and repeat
            cam_matrix[0, 0] = min_x
            cam_matrix[1, 1] = min_y

            # Step 6: Save data to CSV file
            self.save_data()
        print min_x, min_y

        # Determine improved position and rotation
        trans, rot = self.estimate_pose(cam_matrix, distortion_matrix, self.ts_n, training = False)
        self.draw_comparrisson(rot, trans, rot_1, trans_1, vicon_data, p_off)

    def save_data(self):
        camfile = csv.writer(open('camfile.csv', 'w'))
        viconfile = csv.writer(open('viconfile.csv', 'w'))
        errfile = csv.writer(open('errfile.csv', 'w'))

        for j in range(6):
            camfile.writerow(self.camdata[j, :])
            viconfile.writerow(self.vicondata[j, :])
            errfile.writerow(self.errdata[j, :])

    def draw_comparrisson(self, i_rot, i_trans, o_rot, o_trans, vicon_data, p_off):

        titles = ['x', 'y', 'z', 'roll', 'pitch', 'yaw']

        o_p, = plt.plot(o_trans[0, :] - p_off[0], 'r:')
        i_p, = plt.plot(i_trans[0, :] - p_off[0], 'g--')
        v_p, = plt.plot(vicon_data[0, :] * 1.375, 'b_')
        plt.legend([o_p, i_p, v_p], ['Original', 'Improved', 'Vicon'])
        plt.title(titles[0])
        plt.show()

        o_p, = plt.plot(o_trans[1, :] - p_off[1], 'r:')
        i_p, = plt.plot(i_trans[1, :] - p_off[1], 'g--')
        v_p, = plt.plot(vicon_data[1, :], 'b_')
        plt.legend([o_p, i_p, v_p], ['Original', 'Improved', 'Vicon'])
        plt.title(titles[1])
        plt.show()

        o_p, = plt.plot(o_trans[2, :] - p_off[2], 'r:')
        i_p, = plt.plot(i_trans[2, :] - p_off[2], 'g--')
        v_p, = plt.plot(vicon_data[2, :], 'b_')
        plt.legend([o_p, i_p, v_p], ['Original', 'Improved', 'Vicon'])
        plt.title(titles[2])
        plt.show()

        o_p, = plt.plot(o_rot[0, :] - p_off[3], 'r:')
        i_p, = plt.plot(i_rot[0, :] - p_off[3], 'g--')
        v_p, = plt.plot(vicon_data[3, :], 'b_')
        plt.legend([o_p, i_p, v_p], ['Original', 'Improved', 'Vicon'])
        plt.title(titles[3])
        plt.show()

        o_p, = plt.plot(o_rot[1, :] - p_off[4], 'r:')
        i_p, = plt.plot(i_rot[1, :] - p_off[4], 'g--')
        v_p, = plt.plot(vicon_data[4, :], 'b_')
        plt.legend([o_p, i_p, v_p], ['Original', 'Improved', 'Vicon'])
        plt.title(titles[4])
        plt.show()

        o_p, = plt.plot(o_rot[2, :] - p_off[5], 'r:')
        i_p, = plt.plot(i_rot[2, :] - p_off[5], 'g--')
        v_p, = plt.plot(vicon_data[5, :], 'b_')
        plt.legend([o_p, i_p, v_p], ['Original', 'Improved', 'Vicon'])
        plt.title(titles[5])
        plt.show()

    def draw(self, img, corners, imgpts):
        corner = tuple(corners[0].ravel())

        cv2.line(img, corner, tuple(imgpts[0].ravel()), (255, 0, 0), 5)
        cv2.line(img, corner, tuple(imgpts[1].ravel()), (0, 255, 0), 5)
        cv2.line(img, corner, tuple(imgpts[2].ravel()), (0, 0, 255), 5)

        return img

    def find_pos(self, tvecs):
        return [15 * tvecs[0, 0], 15 * tvecs[1, 0], 15 * tvecs[2, 0]]
        #Return in mm 
        #return [100*tvecs[0, 0], 100*tvecs[1, 0], 100*tvecs[2, 0]]
        return [tvecs[0, 0], tvecs[1, 0], tvecs[2, 0]]

    def find_euler_angles(self, rvecs):
        rmat = cv2.Rodrigues(rvecs)[0]

        yaw = math.atan(rmat[1, 0] / rmat[0, 0])
        pitch = math.atan(-rmat[2, 0] / math.sqrt(rmat[2, 1] ** 2 + rmat[2, 2] ** 2))
        roll = math.atan(rmat[2, 1] / rmat[2, 2])

        return [math.degrees(roll), math.degrees(pitch), math.degrees(yaw)]

if __name__ == '__main__':
    ef = ErrFinder()
    ef.main()
