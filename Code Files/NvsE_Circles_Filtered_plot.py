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

#  PLOT : TAKING NORTHING AND EASTING
circle_utm_north = df_gps['UTM Northing'][67:177].to_numpy()
circle_utm_east = df_gps['UTM Easting'][67:177].to_numpy()
sns.scatterplot(x=circle_utm_east, y=circle_utm_north, color='red', s=20)
plt.xlabel('UTM_EASTING')
plt.ylabel('UTM_NORTHING')
plt.show()

#  PLOT : TAKING LATITUDE AND LONGITUDE
circle_utm_lat = df_gps['Latitude'][67:177].to_numpy()
circle_utm_lon = df_gps['Longitude'][67:177].to_numpy()
sns.scatterplot(x=circle_utm_lat, y=circle_utm_lon, color='blue', s=20)
plt.xlabel('UTM_EASTING')
plt.ylabel('UTM_NORTHING')
plt.show()

"""
The result of this code gives me the plot
rows I need for the circlular data
REQD ROWS : 67 TO 177
"""