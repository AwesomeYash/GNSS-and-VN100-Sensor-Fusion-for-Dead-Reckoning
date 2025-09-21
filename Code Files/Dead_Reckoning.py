import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
import scipy.signal
from scipy.signal import butter,filtfilt, detrend

# Load IMU and GPS CSV files
imu_data = pd.read_csv('trip_imu.csv')
gps_data = pd.read_csv('trip_gps.csv')

# Combine seconds and nanoseconds into a single time column for both IMU and GPS
imu_data['time'] = imu_data['seconds'] + imu_data['nanoseconds'] / 1e9
gps_data['time'] = gps_data['seconds'] + gps_data['nanoseconds'] / 1e9

# Calculate IMU-based velocity by integrating Linear.x (forward acceleration) over time
linear_acc_x = imu_data['Linear.x'].to_numpy()
imu_time = imu_data['time'].to_numpy()

# Remove mean acceleration to correct for any static offset
linear_acc_x -= np.mean(linear_acc_x)

# Apply a low-pass filter to remove high-frequency noise
def low_pass_filter(data, cutoff_freq, sample_rate):
    nyquist = 0.5 * sample_rate
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(1, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

sample_rate = 1 / np.mean(np.diff(imu_time))  # Calculate sample rate from time data
cutoff_frequency = 0.05  # Adjust this cutoff frequency as needed
linear_acc_x = low_pass_filter(linear_acc_x, cutoff_frequency, sample_rate)

# Integrate forward acceleration to get forward velocity (IMU-based velocity)
forward_velocity_imu = scipy.integrate.cumulative_trapezoid(linear_acc_x, imu_time, initial=0)

# Calculate GPS-based velocity from UTM coordinates
utm_easting = gps_data['UTM Easting'].to_numpy()
utm_northing = gps_data['UTM Northing'].to_numpy()
gps_time = gps_data['time'].to_numpy()

# Compute distance between consecutive UTM coordinates
distance = np.sqrt(np.diff(utm_easting)**2 + np.diff(utm_northing)**2)
gps_velocity = distance / np.diff(gps_time)

# Adjust GPS time to match GPS velocity array length
gps_time = gps_time[1:]
fs = 40
time = np.arange(0, len(imu_data) / fs, 1/fs)

# Apply adjustments for the second plot (Zero out negative velocities only)
adjusted_velocity_imu = np.copy(forward_velocity_imu)
adjusted_velocity_imu[adjusted_velocity_imu < 0] = 0

# Dead Reckoning Plots
# Displacement from adjusted IMU 
displacement_imu = scipy.integrate.cumulative_trapezoid(adjusted_velocity_imu, imu_time, initial=0)
displacement_gps = scipy.integrate.cumulative_trapezoid(gps_velocity, gps_time, initial=0)

# Yaw Calibration and Correction
xmag_complete = imu_data['Magnetic.x'].to_numpy()
ymag_complete = imu_data['Magnetic.y'].to_numpy()
zmag_complete = imu_data['Magnetic.z'].to_numpy()

# Using the calibrations from the hard and soft iron corrections
centre = [2.5303002156712467e-07, -1.0219295858871122e-06]  
phi = 0.327 + 1.5
scale = 1.1032166386256146

R = np.array([[np.cos(phi), np.sin(phi)], [-np.sin(phi), np.cos(phi)]])
R1 = np.array([[np.cos(-phi), np.sin(-phi)], [-np.sin(-phi), np.cos(-phi)]])

new_x_magc = []    
new_y_magc = []

for i in range(len(xmag_complete)):
    an2 = R1 @ [xmag_complete[i] - centre[0], ymag_complete[i] - centre[1]]
    #an2 = R1 @ [an1[0], an1[1]]

    new_x_magc.append(an2[0])
    new_y_magc.append(an2[1])

cal_yaw = np.arctan2(new_y_magc, new_x_magc)

# Rotate forward velocity to get Easting and Northing components
Ve = adjusted_velocity_imu * np.cos(cal_yaw)
Vn = adjusted_velocity_imu * np.sin(cal_yaw) 

# Integrate to get the trajectory
trajectory_easting = scipy.integrate.cumulative_trapezoid(Ve * scale, imu_time, initial=0)
trajectory_northing = scipy.integrate.cumulative_trapezoid(Vn * scale, imu_time, initial=0)

# Align starting points to (0,0)
# Subtract the initial position to align both trajectories at the start
trajectory_easting -= trajectory_easting[0]
trajectory_northing -= trajectory_northing[0]

# Align GPS trajectory
gps_easting = utm_easting - utm_easting[0]
gps_northing = utm_northing - utm_northing[0]

# Plot the trajectories
plt.figure(figsize=(12, 6))
#plt.plot(gps_easting, gps_northing, label='GPS Trajectory', color='orange')
plt.plot(trajectory_easting, trajectory_northing, label='IMU Trajectory', color='blue')
plt.title('Trajectory Comparison: IMU vs GPS')
plt.xlabel('Easting (m)')
plt.ylabel('Northing (m)')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(gps_easting, gps_northing, label='GPS Trajectory', color='orange')
plt.plot(trajectory_easting, trajectory_northing, label='IMU Trajectory', color='blue')
plt.title('Trajectory Comparison: IMU vs GPS')
plt.xlabel('Easting (m)')
plt.ylabel('Northing (m)')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(gps_easting, gps_northing, label='GPS Trajectory', color='orange')
plt.plot(trajectory_easting, trajectory_northing, label='IMU Trajectory', color='blue')
plt.title('Trajectory Comparison: IMU vs GPS')
plt.xlabel('Easting (m)')
plt.ylabel('Northing (m)')
plt.legend()
plt.grid(True)
plt.show()
