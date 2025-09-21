# IMPORTING HEADER FILES AND PACKAGES
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy import linalg, integrate
from scipy.signal import butter,filtfilt, detrend
import numpy as np
import math
import scipy

# IMPORTING THE .csv FILES
df_gps_trip = pd.read_csv(r"C:\Users\yashr\OneDrive\Desktop\NEU\Semester 1\RSN\Lab 4\trip_gps.csv",sep=",")
df_imu_trip = pd.read_csv(r"C:\Users\yashr\OneDrive\Desktop\NEU\Semester 1\RSN\Lab 4\trip_imu.csv",sep=",")

fs = 40
time = np.arange(0, len(df_imu_trip) / fs, 1/fs)

roll_original_gyro = df_imu_trip["Roll"].to_numpy()
roll_original_gyro = np.radians(roll_original_gyro)
pitch_original_gyro = df_imu_trip["Pitch"].to_numpy()
pitch_original_gyro = np.radians(pitch_original_gyro)
yaw_original_gyro = df_imu_trip["Yaw"].to_numpy()
yaw_original_gyro = np.radians(yaw_original_gyro)


#Velocity
    #Forward Velocity Estimated from GPS
#fs_gps_trip = 10
#time_gps = np.arange(0, len(df_gps_trip) / fs_gps_trip, 1/fs_gps_trip)

start3 = 0
start3_gps = start3//10
start3_gps_time = start3_gps//2
fs_gps = 10
dt_gps = 1/fs_gps
time_gps = np.arange(0, len(df_gps_trip) / fs_gps, dt_gps)

dist_x = df_gps_trip['UTM Easting']
dist_y = df_gps_trip['UTM Northing']

vel_x = np.diff(dist_x)/(dt_gps)
vel_y = np.diff(dist_y)/(dt_gps)
vel_x = np.append(vel_x,vel_x[-1])
vel_y = np.append(vel_y,vel_y[-1])
vel_forward_est_gps = np.sqrt(vel_x**2 + vel_y**2)

#sns.lineplot(y = vel_forward_est_gps, x = time_gps, label="vel_forward_est_gps")
#plt.show()


#Acceleration
forward_acc = df_imu_trip['Linear.x'].to_numpy()
sns.lineplot(y = forward_acc,x = time, label = 'forward_acc')
#forward_acc = forward_acc*np.cos(pitch_original_gyro) + forward_acc*np.sin(pitch_original_gyro)
new_forward_acc = forward_acc.copy()


lp_fc = 0.2
order = 4
nyq = 0.5 * fs
lp_fc_norm = lp_fc / nyq
b, a = butter(order, lp_fc_norm, btype='lowpass')
forward_acc = filtfilt(b, a, forward_acc)
sns.lineplot(y = forward_acc,x = time, label = 'Forward_Accel_LPF')
plt.show()

new_forward_acc = detrend(new_forward_acc)
sns.lineplot(y = new_forward_acc, x = time, label = 'forward_acc_after_bias removal')
sns.lineplot(y = forward_acc, x = time, label = 'forward_acc_before_bias removal')
plt.grid()
plt.show()


dt = 1/fs
new_forward_acc = new_forward_acc - new_forward_acc[0]
forward_velocity_after = np.zeros_like(new_forward_acc)
forward_velocity_after[0] = new_forward_acc[0]
forward_velocity_after[1:] = scipy.integrate.cumulative_trapezoid(new_forward_acc, dx=dt)

forward_velocity_before = np.zeros_like(forward_acc)
forward_velocity_before[0] = forward_acc[0]
forward_velocity_before[1:] = scipy.integrate.cumulative_trapezoid(forward_acc, dx=dt)

forward_velocity_after = detrend(forward_velocity_after)
sns.lineplot(y = forward_velocity_before, x = time, label = 'before_adjustment')
sns.lineplot(y = forward_velocity_after, x = time, label = 'after_adjustment')
#sns.lineplot(y = vel_forward_est_gps,x = time_gps, label="vel_forward_est_gps")
#sns.lineplot(y = new_forward_acc, x = time, label = 'forward_acc')

plt.xlabel('Time (sec)')
plt.title("Forward Velocity before vs after Adjustment")
plt.ylabel("Velocity m/s")
plt.show()

'''
#Direct 
detrend_vel = detrend(forward_velocity_after)

sns.lineplot(y = detrend_vel,x = time, label = 'after_adjustment_notch')
sns.lineplot(y = forward_velocity_after,x = time, label = 'after_adjustment')
sns.lineplot(y = vel_forward_est_gps,x = time_gps,label="vel_forward_est_gps")
plt.xlabel('Time (sec)')
plt.title("Forward Velocity before vs after Adjustment")
plt.ylabel("Velocity m/s")
plt.show()
'''

forward_vel_final = forward_velocity_after

#Method1

#forward_vel_final = detrend_vel.copy()

for i in range(len(forward_vel_final)):
    if(np.abs(new_forward_acc[i] - new_forward_acc[i-1]) < 0.01 and abs(new_forward_acc[i-1]) <0.0005):
        forward_vel_final[i-1] = 0
    elif (forward_vel_final[i] < 0):
        forward_vel_final[i:] = forward_vel_final[i:] - forward_vel_final[i]

sns.lineplot(y = forward_vel_final,x = time[start3:], label = 'after_adjustment')
#seaborn.lineplot(y = detrend_vel,x = time[start3:], label = 'after_adjustment')
sns.lineplot(y = vel_forward_est_gps,x = time_gps[start3_gps_time:],label="vel_forward_est_gps")
#seaborn.lineplot(y = new_forward_acc,x = time[start3:], label = 'forward_acc')


plt.xlabel('Time (sec)')
plt.title("Forward Velocity before vs after Adjustment")
plt.ylabel("Velocity m/s")
plt.grid()
plt.show()


# DEAD RECKONING!!
distance_imu = np.zeros_like(forward_vel_final)
distance_imu[0] = forward_vel_final[0]
distance_imu[1:] = scipy.integrate.cumulative_trapezoid(forward_vel_final, dx=dt)
sns.lineplot(y = distance_imu,x = time, label = 'imu estimate')

distance_gps = np.zeros_like(vel_forward_est_gps)
distance_gps[0] = vel_forward_est_gps[0]
distance_gps[1:] = scipy.integrate.cumulative_trapezoid(vel_forward_est_gps, dx=dt_gps)
sns.lineplot(y = distance_gps, x = time_gps, label = 'gps_distance')
plt.grid()
plt.show()

Y_dot_dot = df_imu_trip["Linear.y"] - df_imu_trip["Linear.y"][0]
X_dot = forward_vel_final
W = df_imu_trip["Angular.z"]
wX_dot = W * X_dot

lp_fc = 0.2
order = 4
nyq = 0.5 * fs
lp_fc_norm = lp_fc / nyq
b, a = butter(order, lp_fc_norm, btype='lowpass')
Y_dot_dot_filtered = filtfilt(b, a, Y_dot_dot)

sns.lineplot(y = wX_dot,x = time,label = "w.X_dot")
sns.lineplot(y = detrend(Y_dot_dot_filtered), x = time, label = "Y_dot_dot")

plt.xlabel('Time (sec)')
plt.title("Dead Recknoing using imu and gps")
plt.ylabel("Acceleration (m/s^2)")
plt.grid()
plt.show()
Cf_yaw = yaw_original_gyro
#Cf_yaw = complimentary_filtered + 0.4

Vn = np.cos(Cf_yaw)*forward_vel_final  - np.sin(Cf_yaw)*forward_vel_final
Ve = np.sin(Cf_yaw)*forward_vel_final  + np.cos(Cf_yaw)*forward_vel_final

Xe = np.zeros_like(Ve)
Xe[0] = Ve[0]
Xe[1:] = scipy.integrate.cumulative_trapezoid(Ve, dx=dt)
Xe = Xe/2 #scaled by 2
Xn = np.zeros_like(Vn)
Xn[0] = Vn[0]
Xn[1:] = scipy.integrate.cumulative_trapezoid(Vn, dx=dt)
Xn = Xn/2 #

dist_x = df_gps_trip['UTM Easting']
dist_y = df_gps_trip['UTM Northing']
dist_x = dist_x - dist_x[0]
dist_y = dist_y - dist_y[0]

plt.plot(Xn, Xe, label = 'imu_distance')
plt.plot(dist_y, dist_x, label = 'gps_distance')
plt.xlabel('Easting')
plt.title("Path Estimations")
plt.ylabel("Northing")
plt.grid()
plt.show()