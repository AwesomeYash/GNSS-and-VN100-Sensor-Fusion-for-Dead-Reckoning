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
df_gps = pd.read_csv(r"C:\Users\yashr\OneDrive\Desktop\NEU\Semester 1\RSN\Lab 4\circles_gps.csv",sep=",")
df_imu = pd.read_csv(r"C:\Users\yashr\OneDrive\Desktop\NEU\Semester 1\RSN\Lab 4\circles_imu.csv",sep=",")
df_gps_trip = pd.read_csv(r"C:\Users\yashr\OneDrive\Desktop\NEU\Semester 1\RSN\Lab 4\trip_gps.csv",sep=",")
df_imu_trip = pd.read_csv(r"C:\Users\yashr\OneDrive\Desktop\NEU\Semester 1\RSN\Lab 4\trip_imu.csv",sep=",")

# TAKING THE MagX, MagY, MagZ data as array
x_Mag_circle = df_imu['Magnetic.x'][2650:6500] #According to the time
y_Mag_circle = df_imu['Magnetic.y'][2650:6500] #from gps data 
z_Mag_circle = df_imu['Magnetic.z'][2650:6500] 

# PLOTING MagX vs MagY (without corrections)
"""
# PLOTING MagX vs MagY (without corrections)
plt.title("Magnetic X vs Magnetic Y (without corrections)")
plt.xlabel('Magnetometre X field')
plt.ylabel('Magnetometre Y field')
plt.grid()
sns.scatterplot(x=x_Mag_circle, y=y_Mag_circle, color='blue', s=20, alpha = 0.5)
plt.show()
"""

# FUNCTION TO FIND VALUES OF THE ELLIPSE
def fit_ellipse(x, y):
    
    D1 = np.vstack([x**2, x*y, y**2]).T
    D2 = np.vstack([x, y, np.ones(len(x))]).T
    S1 = D1.T @ D1
    S2 = D1.T @ D2
    S3 = D2.T @ D2
    T = -np.linalg.inv(S3) @ S2.T
    M = S1 + S2 @ T
    C = np.array(((0, 0, 2), (0, -1, 0), (2, 0, 0)), dtype=float)
    M = np.linalg.inv(C) @ M

    eigval, eigvec = np.linalg.eig(M)
    con = 4 * eigvec[0]* eigvec[2] - eigvec[1]**2
    ak = eigvec[:, np.nonzero(con > 0)[0]]
    coeffs =  np.concatenate((ak, T @ ak)).ravel()
    
    a = coeffs[0]
    b = coeffs[1] / 2
    c = coeffs[2]
    d = coeffs[3] / 2
    f = coeffs[4] / 2
    g = coeffs[5]

    den = b**2 - a*c
    if den > 0:
        raise ValueError('coeffs do not represent an ellipse: b^2 - 4ac must'
                         ' be negative!')

    # The location of the ellipse centre.
    x0, y0 = (c*d - b*f) / den, (a*f - b*d) / den

    num = 2 * (a*f**2 + c*d**2 + g*b**2 - 2*b*d*f - a*c*g)
    fac = np.sqrt((a - c)**2 + 4*b**2)
    # The semi-major and semi-minor axis lengths (these are not sorted).
    ap = np.sqrt(num / den / (fac - a - c))
    bp = np.sqrt(num / den / (-fac - a - c))

    # Sort the semi-major and semi-minor axis lengths but keep track of
    # the original relative magnitudes of width and height.
    width_gt_height = True
    if ap < bp:
        width_gt_height = False
        ap, bp = bp, ap

    # The eccentricity.
    r = (bp/ap)**2
    if r > 1:
        r = 1/r
    e = np.sqrt(1 - r)

    # The angle of anticlockwise rotation of the major-axis from x-axis.
    if b == 0:
        phi = 0 if a < c else np.pi/2
    else:
        phi = np.arctan((2.*b) / (a - c)) / 2
        if a > c:
            phi += np.pi/2
    if not width_gt_height:
        # Ensure that phi is the angle to rotate to the semi-major axis.
        phi += np.pi/2
    phi = phi % np.pi

    return x0, y0, ap, bp, e, phi

# CALLING FUNCTION TO FIND VALUES 
center_x, centre_y, width, height,ecc, phi = fit_ellipse(x_Mag_circle,y_Mag_circle)
centre = [center_x, centre_y]

# PRINTING THE VALUES FOR FURTHER CORRECTIONS 
"""
# PRINTING THE VALUES FOR FURTHER CORRECTIONS 
print("centre coordinates ", centre)
print("width of the ellispse ",width)
print("height of the ellipse ", height)
print("Roation along z ", phi)
print(width/height)
"""

# PLOT : ELLIPSE UNCORRECTED
""" 
fig = plt.figure(figsize=(7,7))
ax = plt.subplot()
ax.axis('equal')
sns.scatterplot(x=x_Mag_circle, y=y_Mag_circle, color='green', s=30,alpha = 0.7)
ellipse = Ellipse(
        xy=centre, width=2*width, height=2*height, angle= np.rad2deg(phi),
        edgecolor='b', fc='None', lw=2, label='Fit', zorder=2
    )

ax.add_patch(ellipse)
plt.title("Uncorrected Magnetometer Data")
plt.xlabel('Magnetic X')
plt.ylabel('Magnetic Y')
plt.grid()
plt.legend()
plt.show()
"""

#Results
"""
# RESULTS  
centre coordinates : 2.5303002156712467e-07, -1.0219295858871122e-06
width of the ellispse  2.2851345731471604e-05
height of the ellipse  2.0713380247727023e-05
Roation along z  0.3270946652707443 rad
1.1032166386256146 => Almost a circle!

"""

# CORRECTIONS
    #ROTATIONS FOR SOFT IRON CORRECTION
R = np.array([[np.cos(phi), np.sin(phi)], [-np.sin(phi), np.cos(phi)]])
R1 = np.array([[np.cos(-phi), np.sin(-phi)], [-np.sin(-phi), np.cos(-phi)]])
scale  = width/height

new_x_mag_circle=[]
new_y_mag_circle=[]

    #SUBTRACTIONS FOR HARD IRON CORRECTION
for i in range(len(x_Mag_circle)):
    an1 = R@[x_Mag_circle[i+2650] - centre[0], y_Mag_circle[i+2650] - centre[1]]
    an2 = R1@[an1[0]/scale, an1[1]]
    new_x_mag_circle.append(an2[0])
    new_y_mag_circle.append(an2[1])

# PLOTTING CORRECTED AND UNCORRECTED MagX Vs MagY DATA
"""
# PLOTTING CORRECTED AND UNCORRECTED MagX Vs MagY DATA
fig = plt.figure(figsize=(7,7))
ax = plt.subplot()
ax.axis('equal')
sns.scatterplot(x=new_x_mag_circle, y=new_y_mag_circle, label = 'Corrected', color='red', s=20,alpha = 0.5)
sns.scatterplot(x=x_Mag_circle, y=y_Mag_circle, label ='Uncorrected', color='blue', s=20,alpha = 0.5 )
plt.title("Corrected & Uncorrected MagX vs MagY")
plt.xlabel('Magnetometre X field')
plt.ylabel('Magnetometre Y field')
plt.grid()
plt.legend()
plt.show()
"""

# PLOT : CORRECTED ELLIPSE 
"""
x_new = np.array(new_x_mag_circle)
y_new = np.array(new_y_mag_circle)
center_x, centre_y, width, height,ecc, phi = fit_ellipse(x_new,y_new)
centre2 = [center_x, centre_y]
new_scale = width/height

fig = plt.figure(figsize=(7,7))
ax = plt.subplot()
ax.axis('equal')
sns.scatterplot(x=new_x_mag_circle, y=new_y_mag_circle, color='red', s=20,alpha = 0.5)
ellipse = Ellipse(
        xy=centre2, width=2*width, height=2*height, angle= np.rad2deg(phi),
        edgecolor='b', fc='None', lw=2, label='Fit', zorder=2
    )
ax.add_patch(ellipse)
plt.title("Corrected Magnetometer Data")
plt.xlabel('Magnetic X')
plt.ylabel('Magnetic Y')
plt.grid()
plt.legend()
plt.show()
#print(new_scale)    #RESULT : 1.000000000000006 => circle!
"""

# TAKING IN ALL THE DATA TO PLOT THE YAW, RAW AND CORRECTED
xmag_complete = df_imu_trip['Magnetic.x'].to_numpy() #NEW VARIABLES FOR TRIP DATA
ymag_complete = df_imu_trip['Magnetic.y'].to_numpy()
zmag_complete = df_imu_trip['Magnetic.z'].to_numpy()

new_x_magc = []    
new_y_magc = []

# CALIBERATING YAW ACCORDING TO MAGNETIC FEILD DATA
for i in range(len(xmag_complete)):
    an1 = R@[xmag_complete[i] - centre[0], ymag_complete[i] - centre[1]]
    an2 = R1@[an1[0]/scale, an1[1]]

    new_x_magc.append(an2[0])
    new_y_magc.append(an2[1])

plt.figure(figsize=(14, 6))

    # TIME SETTINGS 
fs = 40
time = np.arange(0, len(df_imu_trip) / fs, 1/fs)

# RAW AND CALIBERATED YAW IN NEW ARRAYS
raw_mag_yaw = (np.arctan2(ymag_complete, xmag_complete))
cal_mag_yaw = (np.arctan2(new_y_magc, new_x_magc)) 

# PLOT TIME SERIES OF Mag_Data BEFORE AND AFTER CALIBERATION
"""
#Time Series of Magnetometer
    # Plotting X-magnetometer data
plt.subplot(2, 1, 1)
plt.plot(time, new_x_magc, label='Raw X-Magnetometer', color='blue', alpha=0.7)
plt.plot(time, xmag_complete, label='Calibrated X-Magnetometer', color='red', alpha=0.7)
plt.title('Magnetometer Data Before and After Correction')
plt.xlabel('Time (seconds)')
plt.ylabel('Magnetic Field (X)')
plt.legend()
plt.grid()

    # Plotting Y-magnetometer data
plt.subplot(2, 1, 2)
plt.plot(time, new_y_magc, label='Raw Y-Magnetometer', color='blue', alpha=0.7)
plt.plot(time, ymag_complete, label='Calibrated Y-Magnetometer', color='red', alpha=0.7)
plt.xlabel('Time (seconds)')
plt.ylabel('Magnetic Field (Y)')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
"""

# PLOT: CALIBERATED MAGNETOMETER YAW VS RAW YAW (MAGNETOMETER)
"""
# PLOT: CALIBERATED MAGNETOMETER YAW VS RAW YAW (MAGNETOMETER)
plt.figure(figsize=(7, 7))
plt.xlabel("Time in seconds")
plt.ylabel("Yaw in radians")
plt.plot(time, np.unwrap(raw_mag_yaw), label = "Raw Yaw", color = 'blue', alpha=0.75)
plt.plot(time, np.unwrap(cal_mag_yaw), label = "Caliberated Yaw", color = 'red', alpha=0.75)
plt.title('Raw Yaw V/s Caliberated Yaw')
plt.legend()
plt.show()
"""

yaw = df_imu_trip["Angular.z"].to_numpy()
gyro_yaw = scipy.integrate.cumulative_trapezoid(yaw, time, initial=0)

#PLOT : CALIBERATED MAGNETOMETER YAW V/S YAW INTERGRAGTED FROM GYRO
"""
plt.xlabel("Time in seconds")
plt.ylabel("Yaw in radians")
plt.plot(time, np.unwrap(cal_mag_yaw), label = "Calibrated Magnetometer Yaw", color = 'orange')
#plt.plot(time, np.unwrap(raw_mag_yaw), label = "Raw Magnetometer Yaw", color = 'yellow')
plt.plot(time, np.unwrap(gyro_yaw),  label = "Yaw Integrated from Gyro")
plt.title("Calibrated Magnetometer Yaw & Yaw Integrated from Gyro")
plt.legend()
plt.show()
"""

# LOW PASS FILTER FOR MAGNETOMETER
lp_fc = 0.05
order = 4
nyq = 0.5 * fs
lp_fc_norm = lp_fc / nyq
b, a = butter(order, lp_fc_norm, btype='lowpass')
filtered_magnetometer = filtfilt(b, a, cal_mag_yaw)

# HIGH PASS FILTER FOR GYRO
hp_fc = 0.00001
order = 4
nyq = 0.5 * fs
hp_fc_norm = hp_fc / nyq
b, a = butter(order, hp_fc_norm, btype='highpass')
filtered_yaw_integrated = filtfilt(b, a, gyro_yaw)

# COMPLIMENTARY FILTER (COMBINING BOTH)
alpha = 0.95
complimentary_filtered = alpha*np.unwrap(filtered_yaw_integrated) + (1-alpha)*np.unwrap(filtered_magnetometer)

#CALCULATING YAW FROM ORIENTATION (GYRO ORIGINAL DATA) 
x = df_imu_trip['Orientation.x'].to_numpy()
y = df_imu_trip['Orientation.y'].to_numpy()
z = df_imu_trip['Orientation.z'].to_numpy()
w = df_imu_trip['Orientation.w'].to_numpy()
q = np.array([x,y,z,w])

#yaw_original_gyro = np.unwrap(np.arctan2(2*(q[3]*q[2] + q[0]*q[1]), 1 - 2*(q[1]**2 + q[2]**2)))  #BASICALLY YAW FROM IMU

# LPF vs HPF vs CF
"""
plt.figure(figsize=(10, 10))
#sns.lineplot(y = np.unwrap(gyro_yaw), x = time, label = 'YAW Direct Integration')
#sns.lineplot(y = np.unwrap(yaw_original_gyro), x = time, label = 'YAW FROM IMU')
sns.lineplot(y = np.unwrap(filtered_magnetometer), x = time, label = 'LPF')
sns.lineplot(y = np.unwrap(filtered_yaw_integrated), x = time, label = 'HPF')
sns.lineplot(y = np.unwrap(complimentary_filtered), x = time, label = 'CF')
plt.xlabel("Time in seconds")
plt.ylabel("Yaw in radians")
plt.grid()
plt.legend()
plt.show()
"""

# Yaw Integrated from Gyro vs Complimentary filtered yaw
"""
plt.figure(figsize=(10, 10))
sns.lineplot(y = np.unwrap(gyro_yaw), x = time, label = 'YAW Direct Integration')
sns.lineplot(y = np.unwrap(complimentary_filtered), x = time, label = 'CF')
plt.xlabel("Time in seconds")
plt.ylabel("Yaw in radians")
plt.grid()
plt.legend()
plt.show()
"""