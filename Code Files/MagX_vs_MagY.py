# IMPORTING HEADER FILES AND PACKAGES
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt

# IMPORTING THE .csv FILES
df_gps = pd.read_csv(r"C:\Users\yashr\OneDrive\Desktop\NEU\Semester 1\RSN\Lab 4\circles_gps.csv",sep=",")
df_imu = pd.read_csv(r"C:\Users\yashr\OneDrive\Desktop\NEU\Semester 1\RSN\Lab 4\circles_imu.csv",sep=",")
#df_gps_trip = pd.read_csv(r"C:\Users\yashr\OneDrive\Desktop\NEU\Semester 1\RSN\Lab 4\trip_gps.csv",sep=",")
#df_imu_trip = pd.read_csv(r"C:\Users\yashr\OneDrive\Desktop\NEU\Semester 1\RSN\Lab 4\trip_imu.csv",sep=",")

# TAKING THE MagX, MagY, MagZ data as array
x_Mag_circle = df_imu['Magnetic.x'][2650:6500] #According to the time
y_Mag_circle = df_imu['Magnetic.y'][2650:6500] #from gps data 


# PLOTING MagX vs MagY (without corrections)
plt.xlabel('Magnetometre X field')
plt.ylabel('Magnetometre Y field')
plt.grid()
sns.scatterplot(x=x_Mag_circle, y=y_Mag_circle, color='blue', s=20, alpha = 0.5)
plt.show()


#"""  Trial Hard Iron Corrections 
# CALIBRATION
min_x = min(x_Mag_circle)
max_x = max(x_Mag_circle)
min_y = min(y_Mag_circle)
max_y = max(y_Mag_circle)

# HARD-IRON CALIBRATION
x_axis_Offset = (min_x + max_x)/2.0
y_axis_Offset = (min_y + max_y)/2.0
print("hard-iron x_axis_Offset=", x_axis_Offset)
print("hard-iron y_axis_Offset=", y_axis_Offset)
hard_iron_x = []
p = hard_iron_x.extend(x_Mag_circle - x_axis_Offset)
hard_iron_y = []
q = hard_iron_y.extend(y_Mag_circle - y_axis_Offset)

# PLOTING MagX vs MagY (with hard iron corrections)
plt.xlabel('Magnetometre X field')
plt.ylabel('Magnetometre Y field')
plt.grid()
sns.scatterplot(x=x_Mag_circle, y=y_Mag_circle, color='blue', s=20, alpha = 0.5)
sns.scatterplot(x=hard_iron_x, y=hard_iron_y, color='red', s=20, alpha = 0.5)
plt.show()
#"""