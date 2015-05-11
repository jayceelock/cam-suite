#!/usr/bin/python

import csv
import numpy as np
import matplotlib.pyplot as plt

class ErrorPlotter():

    def __init__(self):
        print 'mem'
        self.err_file = 'errfile.csv'

    def read_error_file(self):

        csv_reader = csv.reader(open(self.err_file, 'r'))       

    def main(self):

        print 'main'

        self.read_error_file()

if __name__ == '__main__':

    ep = ErrorPlotter()
    ep.main()
