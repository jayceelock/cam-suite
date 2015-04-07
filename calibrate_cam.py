import sys
import getopt
import cv2
import glob
import numpy as np

def main(argv):

    try:
        opts, args = getopt.getopt(argv,"h:o:ec:",["hi-def=", "output-file=", "leftor-right"])
    except getopt.GetoptError:
        print 'Usage: calibrate_cam.py  -e <hi-def> -o <outputfile.npz> -c <left or right cam>'
        sys.exit(2)

    img_path = "images/"
    outfile = "_cam_calib_params.npz"
    l_or_r = "left"
    search_size = (6, 4)

    for opt, arg in opts:
        if opt == '-h':
            print 'Usage: calibrate_cam.py  -e <hi-def> -o <outputfile.npz> -c <left or right cam>'
            sys.exit()
        elif opt in ("-o", "--output-file"):
            outfile = arg
        elif opt in ("-c", "--left-or-right"):
            l_or_r = arg
            outfile = l_or_r + outfile
        elif opt in ("-e", "--hi-def"):
            search_size = (5, 4)
            img_path = "hd_" + img_path
            outfile = "hd_" + outfile

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((search_size[0] * search_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:search_size[0], 0:search_size[1]].T.reshape(-1, 2)

    objpoints = []
    imgpoints = []

    outfile = "calib_params/" + outfile

    for img_file in glob.glob(img_path + "*" + l_or_r + "*"):
        img = cv2.imread(img_file, 0)
        #print img_file

        ret, corners = cv2.findChessboardCorners(img, search_size, None)

        if ret:
            print 'Found corners in image ' + img_file
            cv2.cornerSubPix(img, corners, (11, 11), (-1, -1), criteria)
            objpoints.append(objp)
            imgpoints.append(corners)

        else:
            print 'Corners not found in image ' + img_file

    print 'Starting calibration...'
    ret, cam_matrix, distortion_matrix, r_vec, t_vec = cv2.calibrateCamera(objpoints, imgpoints, img.shape[::-1], None, None)
    print 'Success. Saving to file ' + outfile
    np.savez(outfile, ret = ret, cam_matrix = cam_matrix, distortion_matrix = distortion_matrix, r_vec = r_vec, t_vec = t_vec)
    print 'Script success'

if __name__ == '__main__':
    main(sys.argv[1:])