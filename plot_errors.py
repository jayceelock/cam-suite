#!/usr/bin/python

import csv
import numpy as np
import matplotlib.pyplot as plt

class ErrorPlotter():

    def __init__(self):

        self.err_file = 'optimise_focal/errfile.csv'

        self.x = []
        self.y = []
        self.z = []
        self.roll = []
        self.pitch = []
        self.yaw = []

    def read_error_file(self):

        csv_reader = csv.reader(open(self.err_file, 'r'))       
        
        for row in csv_reader:
            print type(row)
            print len(row)
            for elem in row:
                print row



    def main(self):

        print 'main'

        self.read_error_file()

if __name__ == '__main__':

    ep = ErrorPlotter()
    ep.main()
