import csv
import numpy as np
import matplotlib.pyplot as plt

v_x = []
v_y = []
v_z = []
v_roll = []
v_pitch = []
v_yaw = []

index = 0
v_x_t = 0
v_y_t = 0
v_z_t = 0
v_roll_t = 0
v_pitch_t = 0
v_yaw_t = 0

csvfile = csv.reader(open('vicon_data.csv', 'r'))

for row in csvfile:
    v_x_t = v_x_t + float(row[0])
    v_y_t = v_y_t + float(row[1])
    v_z_t = v_z_t + float(row[2])
    v_roll_t = v_roll_t + float(row[3])
    v_pitch_t = v_pitch_t + float(row[4])
    v_yaw_t = v_yaw_t + float(row[5])

    index += 1

    if index == 10:
        v_x.append(v_x_t / 10.0)
        v_y.append(v_y_t / 10.0)
        v_z.append(v_z_t / 10.0)
        v_roll.append(v_roll_t / 10.0)
        v_pitch.append(v_pitch_t / 10.0)
        v_yaw.append(v_yaw_t / 10.0)

        index = 0
        v_x_t = 0
        v_y_t = 0
        v_z_t = 0
        v_roll_t = 0
        v_pitch_t = 0
        v_yaw_t = 0

c_x = []
c_y = []
c_z = []
c_roll = []
c_pitch = []
c_yaw = []

csvfile = csv.reader(open('data/no_epnp_sd_test2.csv', 'r'))

for row in csvfile:
    c_x.append(float(row[3]))
    c_y.append(float(row[4]))
    c_z.append(float(row[5]))
    c_roll.append(float(row[0]))
    c_pitch.append(float(row[1]))
    c_yaw.append(float(row[2]))

c_x = [c_x[i]*-10 + 0 for i in range(0, len(c_x))]
c_y = [c_y[i]*-10 + 711 for i in range(0, len(c_x))]    #711
c_z = [c_z[i]*-10 + 2950 for i in range(0, len(c_x))]   #2950

c_pitch = [c_pitch[i]*-1 + 88 for i in range(0, len(c_x))]
c_yaw = [c_yaw[i] - 90 if c_yaw[i] > 0 else c_yaw[i] + 90 for i in range(0, len(c_x)) ]
#c_yaw = [c_yaw[i] * 1 for i in range(len(c_x))]

e_x = np.asarray(v_x[:2641]) - np.asarray(c_x)
e_x = e_x[~np.isnan(e_x)]
e_y = np.asarray(v_y[:2641]) - np.asarray(c_z)
e_y = e_y[~np.isnan(e_y)]
e_z = np.asarray(v_z[:2641]) - np.asarray(c_y)
e_z = e_z[~np.isnan(e_z)]

e_roll = np.asarray(v_roll[:2641]) - np.asarray(c_yaw)
e_roll = e_roll[~np.isnan(e_roll)]
e_pitch = np.asarray(v_pitch[:2641]) - np.asarray(c_pitch)
e_pitch = e_pitch[~np.isnan(e_pitch)]
e_yaw = np.asarray(v_yaw[:2641]) - np.asarray(c_roll)
e_yaw = e_yaw[~np.isnan(e_yaw)]

# p_x, = plt.plot(e_x)
# p_y, = plt.plot(e_y)
# p_z, = plt.plot(e_z)
# plt.legend([p_x, p_y, p_z], ['X', 'Y', 'Z'])
# plt.show()
#
# p_roll, = plt.plot(e_roll)
# p_pitch, = plt.plot(e_pitch)
# p_yaw, = plt.plot(e_yaw)
# plt.legend([p_roll, p_pitch, p_yaw], ['Roll', 'Pitch', 'Yaw'])
# plt.show()

p_off = np.array([np.mean(e_x), np.mean(e_y), np.mean(e_z), np.mean(e_roll), np.mean(e_pitch), np.mean(e_yaw)]).T
print p_off

p_err_x = e_x - p_off[0]
print p_err_x