import os
import pandas as pd
import xarray as xr
import glob
#matplotlib widget
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow
from datetime import datetime

from Functions_sfcwinds import *

base_dir = "C:/Users/memes/Documents/SIPS/Correction"
base_dir = "/kfs2/projects/sfcwinds/test_case"

os.chdir(base_dir)

correction_era_3 = np.load('corrected_HRRR_Atlanta/correction_era_3.npz')
correction_era_10 = np.load('corrected_HRRR_Atlanta/correction_era_10.npz')
correction_hrrr_3 = np.load('corrected_HRRR_Atlanta/correction_hrrr_3.npz')
correction_hrrr_10 = np.load('corrected_HRRR_Atlanta/correction_hrrr_10.npz')

#print(correction_hrrr_3.files)

correction_era_3_u = correction_era_3['u']
correction_era_3_v = correction_era_3['v']
correction_era_10_u = correction_era_10['u']
correction_era_10_v = correction_era_10['v']

correction_hrrr_3_u = correction_hrrr_3['u']
correction_hrrr_3_v = correction_hrrr_3['v']
correction_hrrr_10_u = correction_hrrr_10['u']
correction_hrrr_10_v = correction_hrrr_10['v']

correction_hrrr_3_wspd = (correction_hrrr_3_u**2+correction_hrrr_3_v**2)**0.5
correction_hrrr_10_wspd = (correction_hrrr_10_u**2+correction_hrrr_10_v**2)**0.5
correction_hrrr_3_wdir = np.degrees(np.arctan2(correction_hrrr_3_u, correction_hrrr_3_v)) +180
correction_hrrr_10_wdir = np.degrees(np.arctan2(correction_hrrr_10_u, correction_hrrr_10_v)) +180

#%% HRRR test case May 2024
HRRR_May_2024 = pd.read_pickle('HRRR_NM_2024-05-01_2024-05-31_3m.pkl')
lats = HRRR_May_2024['latitude'].values
lons = HRRR_May_2024['longitude'].values
hrrr_latitude = np.unique(lats)
hrrr_longitude = np.unique(lons)


#%% Replot corrected HRRR vs obs
obs_SVAN5_May2024 = pd.read_csv("Obs hrrr comparison results May 2024/SVAN5_3m_hrrr_10m_merged.csv")

#obs_SVAN5_May2024_3m = obs_SVAN5_May2024["obs_wspd3"]

#obs_SVAN5_May2024 = obs_SVAN5_May2024.set_index("timestamp")
#obs_SVAN5_May2024_hourly = obs_SVAN5_May2024.resample("1H").mean(numeric_only=True)

# Ensure timestamp is in datetime format
obs_SVAN5_May2024["timestamp"] = pd.to_datetime(obs_SVAN5_May2024["timestamp"])

# Set timestamp as the DataFrame index
obs_SVAN5_May2024 = obs_SVAN5_May2024.set_index("timestamp")

# Now resample to hourly and take the mean
obs_SVAN5_May2024_hourly = obs_SVAN5_May2024.resample("1H").mean(numeric_only=True)

closest_lat = obs_SVAN5_May2024["hrrr_latitude"].iloc[0]
closest_lon = obs_SVAN5_May2024["hrrr_longitude"].iloc[0]


####

#34.4652
#-105.08737

# We need the original HRRR data to get the index of the closest point
hrrr = xr.open_dataset(f'HRRR_NM_2024-05-01_2024-05-31.nc')
hrrr_loc, cl_lat, cl_lon, idx = find_closest_HRRR_loc(hrrr, [closest_lat, closest_lon])

iy, ix = idx[-2], idx[-1]
ny = hrrr.sizes['y']
nx = hrrr.sizes['x']

flat_idx = np.ravel_multi_index((iy, ix), (ny, nx))


# 2D -> 1D flat index
flat_idx = np.ravel_multi_index((iy, ix), (ny, nx))

hrrr_index_svan = flat_idx


####




#%% Replot corrected HRRR vs obs
obs_SLMN5_May2024 = pd.read_csv("Obs hrrr comparison results May 2024/SLMN5_3m_hrrr_10m_merged.csv")

#obs_SLMN5_May2024_3m = obs_SLMN5_May2024["obs_wspd3"]

#obs_SLMN5_May2024 = obs_SLMN5_May2024.set_index("timestamp")
#obs_SLMN5_May2024_hourly = obs_SLMN5_May2024.resample("1H").mean(numeric_only=True)

# Ensure timestamp is in datetime format
obs_SLMN5_May2024["timestamp"] = pd.to_datetime(obs_SLMN5_May2024["timestamp"])

# Set timestamp as the DataFrame index
obs_SLMN5_May2024 = obs_SLMN5_May2024.set_index("timestamp")

# Now resample to hourly and take the mean
obs_SLMN5_May2024_hourly = obs_SLMN5_May2024.resample("1H").mean(numeric_only=True)


closest_lat = obs_SLMN5_May2024["hrrr_latitude"].iloc[0]
closest_lon = obs_SLMN5_May2024["hrrr_longitude"].iloc[0]


####

#34.05012
#-108.439667

# We need the original HRRR data to get the index of the closest point
hrrr_loc, cl_lat, cl_lon, idx = find_closest_HRRR_loc(hrrr, [closest_lat, closest_lon])

iy, ix = idx[-2], idx[-1]
ny = hrrr.sizes['y']
nx = hrrr.sizes['x']

flat_idx = np.ravel_multi_index((iy, ix), (ny, nx))


# 2D -> 1D flat index
flat_idx = np.ravel_multi_index((iy, ix), (ny, nx))

hrrr_index_slmn = flat_idx


####


fig, ax = plt.subplots()
ax.plot(obs_SVAN5_May2024_hourly.index,obs_SVAN5_May2024_hourly["obs_wspd3"],label='SVAN5 3m')
ax.plot(obs_SVAN5_May2024_hourly.index,correction_hrrr_3_wspd[hrrr_index_svan,0:744],label='HRRR corr')
#plt.plot(Obs_station_hourly.index[0:742],HRRR_May_2024_station.windspeed_3m[0:742],label='HRRR',alpha=0.5)
ax.set_xlabel('Date')
ax.set_ylabel('Wind speed (m/s)')
ax.legend(loc='upper right',fontsize=10)
#plt.xticks([0,5,10,15,20,30,60,90,120])
#plt.show()
fig.autofmt_xdate()

fig, ax = plt.subplots()
ax.plot(obs_SLMN5_May2024_hourly.index,obs_SLMN5_May2024_hourly["obs_wspd3"],label='SLMN5 3m')
ax.plot(obs_SLMN5_May2024_hourly.index,correction_hrrr_3_wspd[hrrr_index_slmn,0:744],label='HRRR corr')
#plt.plot(Obs_station_hourly.index[0:742],HRRR_May_2024_station.windspeed_3m[0:742],label='HRRR',alpha=0.5)
ax.set_xlabel('Date')
ax.set_ylabel('Wind speed (m/s)')
ax.legend(loc='upper right',fontsize=10)
#plt.xticks([0,5,10,15,20,30,60,90,120])
#plt.show()
fig.autofmt_xdate()











#%% Load data
Obs_May_2024 = pd.read_pickle("Obs_NM_2024-05-01_2024-05-31_3m_10m.pkl")
Obs_July_2024 = pd.read_pickle("Obs_NM_2024-07-01_2024-07-31_3m_10m.pkl")
Obs_July_2024 = Obs_July_2024.set_index("timestamp")

Obs_May_July_2024 = pd.concat([Obs_May_2024, Obs_July_2024])

# Pick one station (replace with your actual ID, e.g. 'STATION123')
station_id = "CELN5"

# Filter for that station
latitude_station = (
    Obs_May_July_2024.loc[Obs_May_July_2024["stid"] == station_id, ["lat"]]
    .dropna()
    .drop_duplicates()
    .to_numpy()
).ravel()

longitude_station = (
    Obs_May_July_2024.loc[Obs_May_July_2024["stid"] == station_id, ["lon"]]
    .dropna()
    .drop_duplicates()
    .to_numpy()
).ravel()


Obs_station = Obs_May_July_2024[Obs_May_July_2024["stid"] == station_id].copy()

# Resample to hourly (mean of all obs within the hour)
Obs_station_hourly = Obs_station.resample("1H").mean(numeric_only=True)

HRRR_May_2024 = pd.read_pickle('HRRR_NM_2024-05-01_2024-05-31_3m.pkl')
lats = HRRR_May_2024['latitude'].values
lons = HRRR_May_2024['longitude'].values
hrrr_latitude = np.unique(lats)
hrrr_longitude = np.unique(lons)

# Generate timestamps from May 1 to May 31, 2024
timestamps_May_2024_15min = pd.date_range(
    start="2024-05-01 00:00:00",
    end="2024-05-31 23:59:59",
    freq="15min",
    tz="UTC"   # ensures datetime64[ns, UTC]
)


def find_closest_HRRR_loc(hrrr, hrrr_coords_to_use):
    
    """
    hrrr: HRRR xarray
    hrrr_coords_to_use: array -> [lat, lon]
    
    
    
    """

    dist = (hrrr['longitude'].values - hrrr_coords_to_use[1])**2 + (hrrr['latitude'].values - hrrr_coords_to_use[0])**2
    idx = np.unravel_index(np.argmin(dist), dist.shape)
    closest_lat = hrrr.latitude.values[idx]
    closest_lon = hrrr.longitude.values[idx]

    hrrr_loc =hrrr.isel(y=idx[0], x=idx[1])
    
    fsr_val = hrrr_loc['fsr'].values if 'fsr' in hrrr_loc else np.nan

    return hrrr_loc, closest_lat, closest_lon, idx
       

hrrr_loc, idx, closest_lat, closest_lon, fsr_val = find_closest_HRRR_loc(HRRR_May_2024, [latitude_station[0], longitude_station[0]])       



#%%
hrrr_index = np.where(hrrr_latitude == closest_lat)[0]
HRRR_May_2024_sorted = HRRR_May_2024.sort_values(by="longitude").reset_index(drop=True)

tol = 1e-3  # tolerance in degrees (~100 m)
HRRR_May_2024_station = HRRR_May_2024_sorted[
    np.isclose(HRRR_May_2024_sorted["latitude"], closest_lat, atol=tol) &
    np.isclose(HRRR_May_2024_sorted["longitude"], closest_lon, atol=tol)
]


import matplotlib.dates as mdates

ax2 = fig.add_subplot(212)
#ax2 = ax1.twinx()
ax2.scatter(time_vector_start_20Hz_20241028,H1_Iw_z_20241028[:,1], color='#1f77b4',marker='o',s=5,label='H1 $I_w$')
ax2.scatter(time_vector_start_20Hz_20241028,H1_Lwxc_z_20241028[:,1], color='cyan',marker='o',s=5,label='H1 $L_w^x/c$')
ax2.scatter(time_vector_start_20Hz_20241028,H2_Iw_z_20241028[:,1], color='#ff7f0e',marker='s',s=5,label='H2 $I_w$')
ax2.scatter(time_vector_start_20Hz_20241028,H2_Lwxc_z_20241028[:,1], color='gold',marker='s',s=5,label='H2 $L_w^x/c$')
ax2.scatter(time_vector_start_20Hz_20241028,H3_Iw_z_20241028[:,1], color='#2ca02c',marker='d',s=5,label='H3 $I_w$')
ax2.scatter(time_vector_start_20Hz_20241028,H3_Lwxc_z_20241028[:,1], color='lime',marker='d',s=5,label='H3 $L_w^x/c$')
ax2.set_ylabel('$I_w$  $L_w^x/c$')
ax2.tick_params(axis='y')
#ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0,fontsize=8)
hour_formatter = mdates.DateFormatter("%H:%M",tz=time_vector_start_20Hz_20241028.tz)  # %H for 24-hour format
#ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=time_vector_start_20Hz_20241028.tz))
ax2.xaxis.set_major_formatter(hour_formatter)
plt.xlabel("Local time (October 28, 2024)")
plt.ylim(0, 0.5) 
plt.yticks([0,0.1,0.2,0.3,0.4,0.5])  
fig.autofmt_xdate()
#ax1.set_xlim([datetime.date(2024,10,28,14,0), datetime.date(2024,10,28,20,0)])
plt.tight_layout()



fig, ax = plt.subplots()
ax.plot(Obs_station_hourly.index,Obs_station_hourly.windspeed,label='obs (3m)',alpha=0.8)
ax.plot(Obs_station_hourly.index[0:742],HRRR_May_2024_station.windspeed10[0:742],label='HRRR (10m)',alpha=0.5)
ax.legend(loc='upper right',fontsize=10)
ax.set_xlabel('Date')
ax.set_ylabel('Wind speed (m/s) at CELN5, NM')
#plt.xticks([0,5,10,15,20,30,60,90,120])
ax.legend(loc='upper right',fontsize=10)
#plt.show()
#ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d")) 
fig.autofmt_xdate()

#plt.tight_layout()
#plt.savefig("obs num time resolution NM.png")
#plt.xlim([datetime.datetime(2024,8,26,0,0), datetime.datetime(2024,9,10,0,0)])
#plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%YYYY-%MM'))
#plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%MM-%DD"))
#ax.xaxis.set_major_formatter(mdates.DateFormatter("%MM-%DD"))
#fig.autofmt_xdate()
#plt.tight_layout()

plt.figure()
plt.plot(Obs_station_hourly.index,Obs_station_hourly.windspeed_10m,label='obs',alpha=0.8)
plt.plot(Obs_station_hourly.index,correction_hrrr_10_wspd[hrrr_index[0],0:744],label='HRRR corrected',alpha=0.7)
plt.plot(Obs_station_hourly.index[0:742],HRRR_May_2024_station.windspeed10[0:742],label='HRRR',alpha=0.5)
plt.legend(loc='upper right',fontsize=10)
plt.xlabel('Date')
plt.ylabel('Wind speed (m/s) at 10m height (CELN5)')
#plt.xticks([0,5,10,15,20,30,60,90,120])
plt.legend(loc='upper right',fontsize=10)
plt.show()


plt.figure()
plt.plot(Obs_station_hourly.index,Obs_station_hourly.winddirection,label='obs')
plt.plot(Obs_station_hourly.index,correction_hrrr_10_wdir[hrrr_index[0],0:744],label='HRRR corrected')
plt.legend(loc='upper right',fontsize=10)
plt.xlabel('Date')
plt.ylabel('Wind direction (deg) at 10m height (CELN5)')
#plt.xticks([0,5,10,15,20,30,60,90,120])
plt.legend(loc='upper right',fontsize=10)
plt.show()

Diff_obs_hrrr_corr_3m = correction_hrrr_3_wspd[hrrr_index[0],0:744]-Obs_station_hourly.windspeed_3m
Diff_obs_hrrr_3m = HRRR_May_2024_station.windspeed_3m[0:742].values-Obs_station_hourly.windspeed_3m[0:742]

Diff_mean_obs_hrrr_corr_3m = Diff_obs_hrrr_corr_3m.mean()
Diff_mean_obs_hrrr_3m = Diff_obs_hrrr_3m.mean()
Diff_median_obs_hrrr_corr_3m = Diff_obs_hrrr_corr_3m.median()
Diff_median_obs_hrrr_3m = Diff_obs_hrrr_3m.median()
Diff_std_obs_hrrr_corr_3m = Diff_obs_hrrr_corr_3m.std()
Diff_std_obs_hrrr_3m = Diff_obs_hrrr_3m.std()

plt.figure()
plt.hist(Diff_obs_hrrr_3m, bins=np.arange(-10, 10, 0.5), range=(-10, 10), edgecolor='none',alpha=0.5,label='HRRR - obs')
plt.hist(Diff_obs_hrrr_corr_3m, bins=np.arange(-10, 10, 0.5), range=(-10, 10), edgecolor='none',alpha=0.5,label='HRRR corr - obs')
plt.legend(loc='upper right',fontsize=10)
plt.xlabel('Difference in wind speed (m/s) at 3m height (CELN5)')
plt.ylabel('Frequency')
plt.xticks([-10,-8,-6,-4,-2,0,2,4,6,8,10])
plt.legend(loc='upper right',fontsize=10)
plt.show()

Diff_obs_hrrr_corr_10m = correction_hrrr_10_wspd[hrrr_index[0],0:744]-Obs_station_hourly.windspeed_10m
Diff_obs_hrrr_10m = HRRR_May_2024_station.windspeed10[0:742].values-Obs_station_hourly.windspeed_10m[0:742]

Diff_mean_obs_hrrr_corr_10m = Diff_obs_hrrr_corr_10m.mean()
Diff_mean_obs_hrrr_10m = Diff_obs_hrrr_10m.mean()
Diff_median_obs_hrrr_corr_10m = Diff_obs_hrrr_corr_10m.median()
Diff_median_obs_hrrr_10m = Diff_obs_hrrr_10m.median()
Diff_std_obs_hrrr_corr_10m = Diff_obs_hrrr_corr_10m.std()
Diff_std_obs_hrrr_10m = Diff_obs_hrrr_10m.std()

plt.figure()
plt.hist(Diff_obs_hrrr_10m, bins=np.arange(-14, 14, 0.5), range=(-14, 14), edgecolor='none',alpha=0.5,label='HRRR - obs')
plt.hist(Diff_obs_hrrr_corr_10m, bins=np.arange(-14, 14, 0.5), range=(-14, 14), edgecolor='none',alpha=0.5,label='HRRR corr - obs')
plt.legend(loc='upper right',fontsize=10)
plt.xlabel('Difference in wind speed (m/s) at 10m height (CELN5)')
plt.ylabel('Frequency')
plt.xticks([-14,-12,-10,-8,-6,-4,-2,0,2,4,6,8,10,12,14])
plt.legend(loc='upper right',fontsize=10)
plt.show()


