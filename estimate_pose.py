import cv2
import sys
import getopt
import numpy as np
import csv
import math

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

    yaw = math.atan(rmat[1, 0] / rmat[0, 0])
    pitch = math.atan(-rmat[2, 0] / math.sqrt(rmat[2, 1] ** 2 + rmat[2, 2] ** 2))
    roll = math.atan(rmat[2, 1] / rmat[2, 2])

    return (math.degrees(roll), math.degrees(pitch), math.degrees(yaw))

def main(argv):

    try:
        opts, args = getopt.getopt(argv,"hi:ec:",["hi-def=", "input-file=", "leftor-right", "output-file"])
    except getopt.GetoptError:
        print 'Usage: calibrate_cam.py -c <left or right cam> -e <hi-def> -i <inputfile.avi> -o <outfile.csv>'
        sys.exit(2)

    infile = "output.avi"
    outfile = "data.csv"
    calib_file = '_cam_calib_params.npz'
    l_or_r = "left"
    search_size = (5, 4)

    for opt, arg in opts:
        if opt == '-h':
            print 'Usage: calibrate_cam.py -c <left or right cam> -e <hi-def> -i <inputfile.avi> -o <outfile.csv>'
            sys.exit()
        elif opt in ("-i", "--input-file"):
            infile = arg
        elif opt in ("-o", "--output-file"):
            outfile = arg
        elif opt in ("-c", "--left-or-right"):
            l_or_r = arg
            calib_file = l_or_r + calib_file
        elif opt in ("-e", "--hi-def"):
            infile = "hd_" + infile
            calib_file = "hd_" + calib_file

    infile = l_or_r + "_" + infile
    infile = "videos/" + infile
    outfile = "data/" + outfile

    with np.load('calib_params/' + calib_file) as X:
        _, cam_matrix, distortion_matrix, r_vec, _ = [X[i] for i in ('ret', 'cam_matrix', 'distortion_matrix', 'r_vec', 't_vec')]

    axis = np.float32([[3, 0, 0], [0, 3, 0], [0, 0, 3]]).reshape(-1, 3)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((search_size[0] * search_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:search_size[0], 0:search_size[1]].T.reshape(-1, 2)

    objpoints = []
    imgpoints = []

    cap = cv2.VideoCapture(infile)

    csvfile = open(outfile, 'w+')
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['roll', 'pitch', 'yaw', 'x', 'y', 'z'])

    while(cap.isOpened()):
        ret, frame = cap.read()
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(grey, search_size, None)

        if ret:
            cv2.cornerSubPix(grey, corners, (11, 11), (-1, -1), criteria)
            rvecs, tvecs, inliers = cv2.solvePnPRansac(objp, corners, cam_matrix, distortion_matrix, flags = cv2.CV_EPNP)
            imgpts, jacobian = cv2.projectPoints(axis, rvecs, tvecs, cam_matrix, distortion_matrix)

            csvwriter.writerow(find_euler_angles(rvecs) + find_pos(tvecs))

            frame = draw(frame, corners, imgpts)
        else:
            print 'Corners not found in this frame'

        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv[1:])