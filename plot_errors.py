#!/usr/bin/python

import csv
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

class ErrorPlotter():

    def __init__(self):

        self.err_file = 'optimise_focal/errfile.csv'

        self.err_data = []

    def read_error_file(self):

        csv_reader = csv.reader(open(self.err_file, 'r'))       
        
        for row in csv_reader:
            self.err_data.append(row)

        self.x = np.asarray([[float(self.err_data[0][i]) for i in range(len(self.err_data[0]))]]) * 2000.0
        self.y = np.asarray([[float(self.err_data[1][i]) for i in range(len(self.err_data[1]))]]) * 5000.0
        self.z = np.asarray([[float(self.err_data[2][i]) for i in range(len(self.err_data[2]))]]) * 1000.0
        self.roll = np.asarray([[float(self.err_data[3][i]) for i in range(len(self.err_data[3]))]])
        self.pitch = np.asarray([[float(self.err_data[4][i]) for i in range(len(self.err_data[4]))]])
        self.yaw = np.asarray([[float(self.err_data[5][i]) for i in range(len(self.err_data[5]))]])
        
    
    def plot_hist(self):

        freq, bins = np.histogram(self.x, bins = 100)

        width = 0.7 * (bins[1] - bins[0])
        center = (bins[:-1] + bins[1:]) / 2
        plt.bar(center, freq, align='center', width=width)
        plt.title('x')
        plt.show()

        freq, bins = np.histogram(self.y, bins = 100)

        width = 0.7 * (bins[1] - bins[0])
        center = (bins[:-1] + bins[1:]) / 2
        plt.bar(center, freq, align='center', width=width)
        plt.title('y')
        plt.show()

        freq, bins = np.histogram(self.z, bins = 100)

        width = 0.7 * (bins[1] - bins[0])
        center = (bins[:-1] + bins[1:]) / 2
        plt.bar(center, freq, align='center', width=width)
        plt.title('z')
        plt.show()

        freq, bins = np.histogram(self.roll, bins = 100)

        width = 0.7 * (bins[1] - bins[0])
        center = (bins[:-1] + bins[1:]) / 2
        plt.bar(center, freq, align='center', width=width)
        plt.title('Roll')
        plt.show()

        freq, bins = np.histogram(self.pitch, bins = 100)

        width = 0.7 * (bins[1] - bins[0])
        center = (bins[:-1] + bins[1:]) / 2
        plt.bar(center, freq, align='center', width=width)
        plt.title('Pitch')
        plt.show()

        freq, bins = np.histogram(self.yaw, bins = 100)

        width = 0.7 * (bins[1] - bins[0])
        center = (bins[:-1] + bins[1:]) / 2
        plt.bar(center, freq, align='center', width=width)
        plt.title('Yaw')
        plt.show()

    def plot_err_convergence(self):

        dim = []
        for i in range(self.pitch.shape[1] / 90 - 90):
            dim.append(self.yaw[:, i * 90: i * 90 + 90])
        dim = np.asarray(dim)
        print dim.shape

        dim = dim.reshape((dim.shape[0], 90))

        avgs = dim.mean(axis = 1).reshape((dim.shape[0], 1))
        mem = np.zeros((50, 1))

        interval = dim.shape[0] / 50
        for i in range(50):
            mem[i, 0] = avgs[i * interval:i * interval + interval].mean()
        #mem[0, 0] = avgs[0:106, :].mean()
        #mem[1, 0] = avgs[106:212, :].mean()
        #mem[2, 0] = avgs[212:318, :].mean()
        #mem[3, 0] = avgs[318:424, :].mean()
        #mem[4, 0] = avgs[424:530, :].mean()
        #mem[5, 0] = avgs[530:, :].mean()

        plt.plot(mem)
        plt.show()
        print mem
        #print avgs
        #plt.plot(avgs)
        #plt.show()

    def is_norm_dist(self):
        
        chi, p = stats.normaltest(self.x)
        print chi, p


    def main(self):

        print 'main'

        self.read_error_file()
        #self.is_norm_dist()

        err_mat = np.concatenate((self.x, self.y))
        err_mat = np.concatenate((err_mat, self.z))
        err_mat = np.concatenate((err_mat, self.roll))
        err_mat = np.concatenate((err_mat, self.pitch))
        err_mat = np.concatenate((err_mat, self.yaw))

        self.plot_err_convergence()

        #print np.cov(err_mat)
        #self.plot_hist()

if __name__ == '__main__':

    ep = ErrorPlotter()
    ep.main()
