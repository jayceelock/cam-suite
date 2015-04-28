import csv
import matplotlib.pyplot as plt

a = []

#reader = csv.reader(open('camfile.csv', 'r'))

filename = open('camfile.csv', 'r')

rows = filename.read().split('\n')

x = rows[0].split(',')
x = [float(i) for i in x]

T = 30
print len(x) / T
for i in range(len(x) / T):
#	print len(x[T*i:T + T*i])
	plt.plot(x[T*i:(T + T*i)])

plt.show()

