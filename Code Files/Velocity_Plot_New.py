import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
import math

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

# Plot both velocity estimates (Original IMU-based and GPS-based velocities)
plt.figure(figsize=(12, 6))
plt.plot(imu_time, forward_velocity_imu, label='IMU-based Velocity (Original)', color='blue')
plt.plot(gps_time, gps_velocity, label='GPS-based Velocity', color='orange')
plt.title('Forward Velocity from IMU and GPS')
plt.xlabel('Time (seconds)')
plt.ylabel('Velocity (m/s)')
plt.legend()
plt.grid(True)
plt.show()

# Apply adjustments for the second plot (Zero out negative velocities only)
# Zero out any negative velocities in IMU-based velocity
adjusted_velocity_imu = np.copy(forward_velocity_imu)
adjusted_velocity_imu[adjusted_velocity_imu < 0] = 0

# Plot the corrected IMU velocity with GPS velocity (Adjusted Plot without filter)
plt.figure(figsize=(12, 6))
plt.plot(imu_time, adjusted_velocity_imu, label='Corrected IMU-based Velocity (No Filter)', color='green')
plt.plot(gps_time, gps_velocity, label='GPS-based Velocity', color='orange')
plt.title('Corrected IMU and GPS Velocity Comparison')
plt.xlabel('Time (seconds)')
plt.ylabel('Velocity (m/s)')
plt.legend()
plt.grid(True)
plt.show()