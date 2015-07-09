#!/usr/bin/python

import cv2
import cv2.cv as cv
import sys
import getopt

from subprocess import call

def main(argv):

    try:
        opts, args = getopt.getopt(argv,"hd:o:ec:",["device=","hi-def=", "output-file=", "left-or-right"])
    except getopt.GetoptError:
        print 'Usage: record_video.py -d <cam-number> -e <hi-def> -o <outputfile.avi> -c <left or right>'
        sys.exit(2)

    high_def = False
    cam_nbr = 0
    outfile = "output.avi"
    l_or_r = 'left'

    for opt, arg in opts:
        if opt == '-h':
            print 'Usage: record_video.py -d <cam-number> -e <hi-def> -o <outputfile.avi> -c <left or right>'
            sys.exit()
        elif opt in ("-d", "--device"):
            cam_nbr = int(arg)
        elif opt in ("-o", "--output-file"):
            outfile = arg
        elif opt in ("-e", "--hi-def"):
            high_def = True
        elif opt in ("-c", "--left-or-right"):
            l_or_r = arg

    if high_def:
        outfile = "hd_" + l_or_r + "_" + outfile
    else:
        outfile =  l_or_r + "_" + outfile
    outfile = "../videos/" + outfile

    try:
        cap = cv2.VideoCapture(cam_nbr)
    except:
        print ' '
    if not cap.isOpened():
        print 'Invalid camera selection'
        sys.exit(2)

    print 'Disabling auto-focus and zoom...'
    try:
        call(["uvcdynctrl", "-d", "video" + str(cam_nbr), "-s", "Focus, Auto", "0"])
        call(["uvcdynctrl", "-d", "video" + str(cam_nbr), "-s", "Zoom, Absolute", "0"])
        print "Success."
    except:
        print 'Error: please check that your camera supports these settings and that uvcdynctrl is installed.'
        sys.exit(2)

    if high_def:
        cap.set(cv.CV_CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv.CV_CAP_PROP_FRAME_WIDTH,  1280)
    else:
        cap.set(cv.CV_CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv.CV_CAP_PROP_FRAME_WIDTH,  640)

    # Define the codec and create VideoWriter object
    fourCC = cv.CV_FOURCC(*'XVID')
    out = cv2.VideoWriter(outfile, fourCC, 30.0, (int(cap.get(cv.CV_CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CV_CAP_PROP_FRAME_HEIGHT))))

    while(cap.isOpened()):
        ret, frame = cap.read()

        if ret:
            out.write(frame)
            cv2.imshow('frame',frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release everything if job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv[1:])
