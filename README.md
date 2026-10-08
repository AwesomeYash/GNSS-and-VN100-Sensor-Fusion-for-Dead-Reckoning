# GNSS and VN-100 Sensor Fusion for Dead Reckoning

Lab project for Robotics Sensing and Navigation (EECE 5554, Northeastern University): calibrate a VectorNav VN-100 magnetometer, estimate heading with a complementary filter, estimate forward velocity, and dead-reckon a trajectory from IMU data, comparing each result with a GNSS (GPS) track. The report is [`Lab4-Report.pdf`](Lab4-Report.pdf) and the assignment handout is [`LAB4_Fall_2024.pdf`](LAB4_Fall_2024.pdf).

## Pipeline

| Stage | What it does | Script |
|---|---|---|
| 1. Magnetometer calibration | Fits an ellipse to magnetometer X-Y data from driving in circles, then removes hard-iron (centre offset) and soft-iron (rotation and axis scaling) distortion | `Hard_Soft_Iron_Corrections.py` |
| 2. Heading | Yaw from the calibrated magnetometer, gyro-integrated yaw, and a complementary filter (low-pass on magnetometer yaw, high-pass on gyro yaw, weight 0.95) | `Yaw_Corrections.py`, `Magnetometer_VS_Gyro.py`, `Lp_Hp_Cp_Filter.py` |
| 3. Forward velocity | Integrates forward acceleration (mean removed, filtered, negatives clipped) and compares with velocity from GPS UTM positions | `Velocity_Plot.py`, `Velocity_Plot_New.py` |
| 4. Dead reckoning | Rotates the IMU forward velocity into easting/northing with the calibrated yaw and integrates to a trajectory; overlays the GPS track | `Dead_Reckoning.py` |

Other scripts: `MagX_vs_MagY.py`, `NvsE_Circles_Filtered_plot.py`, `Final_analysis_script.py` (plots).

## Data

`Sensor Data/` contains two recordings as ROS 2 bags (`Raw Data/circles1`, `Raw Data/trip1`) and their CSV exports:

| File | Content |
|---|---|
| `circles_imu.csv`, `circles_gps.csv` | Driving in circles (magnetometer calibration) |
| `trip_imu.csv`, `trip_gps.csv` | A drive of about 464 s: 12,448 IMU samples (mean rate about 26.8 Hz) and 464 GPS fixes (1 Hz) |

The data were recorded by a teammate (Varun Raghavendra, per the report) and shared with the group.

## Calibration values (kept in `Hard_Soft_Iron_Corrections.py`)

| Quantity | Value |
|---|---|
| Ellipse centre | (2.53e-07, -1.02e-06) T |
| Semi-axes | 2.285e-05 T and 2.071e-05 T (ratio 1.1032) |
| Rotation | 0.327 rad |

## Figures

Result plots are in `Results/Graphs/` (yaw, velocity and filter plots) and `Results/Maps/` (ellipse fits and North-vs-East plots), for example:

- `Results/Maps/Ellipse_Visualized_Corrected.png` and `Results/Maps/H&S_Iron_CorrectionsVsOriginal.png`
- `Results/Graphs/LPF_vs_HPF_vs_CF_vs_Raw.png`
- `Results/Graphs/Forward_vel_imu_gps_corrected_comparison.png`
- `Results/Maps/NorthingVsEasting_Trip.png`

The IMU-versus-GPS trajectory overlay is Figure 13 in the report.

## Requirements and how to run

Python 3 with `numpy`, `pandas`, `scipy`, `matplotlib` and `seaborn`.

- `Dead_Reckoning.py` reads `trip_imu.csv` and `trip_gps.csv` from the working directory: copy the two files from `Sensor Data/` next to the script and run `python Dead_Reckoning.py`.
- The other scripts read the CSV files from an absolute Windows path (`C:\Users\yashr\OneDrive\Desktop\NEU\Semester 1\RSN\Lab 4\...`). Change the `pd.read_csv(...)` paths to `Sensor Data/<file>.csv` before running them.

## Results

Re-running the committed data (computed in a repository audit, not stated in the report):

- GPS path length of the trip is about 1,858 m (median speed 4.2 m/s, net start-to-end displacement about 28 m).
- The integrated IMU forward distance is about 1,355 m with the 0.05 Hz low-pass used in the code, roughly 70 % of the GPS distance before scaling.
- The report shows the dead-reckoned IMU path (after a heading offset of 0.327 + 1.5 rad and a scale of 1.1032) overlapping the GPS track qualitatively, with visible divergence; no numeric trajectory error is reported.

## Known issues

- The report says the acceleration low-pass cutoff is 10 Hz; `Dead_Reckoning.py` uses 0.05 Hz.
- The report says the complementary filter favours the magnetometer; the code computes `0.95 * gyro_highpassed + 0.05 * mag_lowpassed`, which favours the gyro. The 1e-5 Hz gyro high-pass has almost no effect on a 464 s recording.
- The value 1.1032 is the magnetometer ellipse axis ratio, and it is reused as a velocity scale in dead reckoning; the extra 1.5 rad heading offset was tuned by hand.
- The code assumes a 40 Hz IMU rate in some plot time axes, but the logged data average about 26.8 Hz.
- Several scripts repeat the same calibration block; the bonus question in the report is unanswered; absolute paths need editing.

## Credits

Assignment: EECE 5554 Lab 4 (instructor handout included). Data collection by a teammate (see above). `fit_ellipse` in `Hard_Soft_Iron_Corrections.py` closely resembles a widely circulated public implementation of the direct least-squares ellipse fit; add a source link if it was adapted.

## License

MIT License (see `LICENSE`).
