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

        self.x = np.asarray([[float(self.err_data[0][i]) for i in range(len(self.err_data[0]))]]) * 1500.0
        self.y = np.asarray([[float(self.err_data[1][i]) for i in range(len(self.err_data[1]))]]) * 3000.0
        self.z = np.asarray([[float(self.err_data[2][i]) for i in range(len(self.err_data[2]))]]) * 1000.0
        self.roll = np.asarray([[float(self.err_data[3][i]) for i in range(len(self.err_data[3]))]])
        self.pitch = np.asarray([[float(self.err_data[4][i]) for i in range(len(self.err_data[4]))]])
        self.yaw = np.asarray([[float(self.err_data[5][i]) for i in range(len(self.err_data[5]))]])
        
    
    def plot(self, data, tit):
        freq, bins = np.histogram(data, bins = 100)
        
        std = np.std(data)
        dist_range = np.arange(bins[0], bins[-1], 0.1)
        dist = stats.norm.pdf(dist_range, 0, std)
        
        plt.plot(dist_range, dist / float(max(dist)))
        
        width = 0.7 * (bins[1] - bins[0])
        center = (bins[:-1] + bins[1:]) / 2
        
        plt.bar(center, freq / float(max(freq[len(freq) / 4:len(freq) / 4 * 3])), align='center', width=width)
        plt.title(tit)
        
        plt.show()

    def plot_hist(self):
        
        self.plot(self.x, 'x')
        self.plot(self.y, 'y')
        self.plot(self.z, 'z')
        self.plot(self.roll, 'roll')
        self.plot(self.pitch, 'pitch')
        self.plot(self.yaw, 'yaw')

        #freq, bins = np.histogram(self.x, bins = 100)

        #width = 0.7 * (bins[1] - bins[0])
        #center = (bins[:-1] + bins[1:]) / 2
        #plt.bar(center, freq, align='center', width=width)
        #plt.title('x')
        #plt.show()

        #freq, bins = np.histogram(self.y, bins = 100)

        #width = 0.7 * (bins[1] - bins[0])
        #center = (bins[:-1] + bins[1:]) / 2
        #plt.bar(center, freq, align='center', width=width)
        #plt.title('y')
        #plt.show()

        #freq, bins = np.histogram(self.z, bins = 100)

        #width = 0.7 * (bins[1] - bins[0])
        #center = (bins[:-1] + bins[1:]) / 2
        #plt.bar(center, freq, align='center', width=width)
        #plt.title('z')
        #plt.show()

        #freq, bins = np.histogram(self.roll, bins = 100)

        #width = 0.7 * (bins[1] - bins[0])
        #center = (bins[:-1] + bins[1:]) / 2
        #plt.bar(center, freq, align='center', width=width)
        #plt.title('Roll')
        #plt.show()

        #freq, bins = np.histogram(self.pitch, bins = 100)

        #width = 0.7 * (bins[1] - bins[0])
        #center = (bins[:-1] + bins[1:]) / 2
        #plt.bar(center, freq, align='center', width=width)
        #plt.title('Pitch')
        #plt.show()

        #freq, bins = np.histogram(self.yaw, bins = 100)

        #width = 0.7 * (bins[1] - bins[0])
        #center = (bins[:-1] + bins[1:]) / 2
        #plt.bar(center, freq, align='center', width=width)
        #plt.title('Yaw')
        #plt.show()

    def plot_err_convergence(self, data, tit):

        dim = []
        for i in range(data.shape[1] / 90 - 90):
            dim.append(data[:, i * 90: i * 90 + 90])
        dim = np.asarray(dim)

        dim = dim.reshape((dim.shape[0], 90))

        avgs = dim.mean(axis = 1).reshape((dim.shape[0], 1))
        mem = np.zeros((50, 1))

        interval = dim.shape[0] / 50 
        for i in range(50):
            mem[i, 0] = avgs[i * interval:i * interval + interval].mean()

        plt.plot(mem)
        plt.title(tit)
        plt.show()
        print mem

    def is_norm_dist(self):
        
        chi, p = stats.normaltest(self.x)
        print chi, p


    def main(self):

        self.read_error_file()
        #self.is_norm_dist()

        err_mat = np.concatenate((self.x, self.y))
        err_mat = np.concatenate((err_mat, self.z))
        err_mat = np.concatenate((err_mat, self.roll))
        err_mat = np.concatenate((err_mat, self.pitch))
        err_mat = np.concatenate((err_mat, self.yaw))

        self.plot_err_convergence(self.x, 'x')
        self.plot_err_convergence(self.y, 'y')
        self.plot_err_convergence(self.z, 'z')
        self.plot_err_convergence(self.roll, 'roll')
        self.plot_err_convergence(self.pitch, 'pitch')
        self.plot_err_convergence(self.yaw, 'yaw')

        print np.cov(err_mat)
        self.plot_hist()

if __name__ == '__main__':

    ep = ErrorPlotter()
    ep.main()
