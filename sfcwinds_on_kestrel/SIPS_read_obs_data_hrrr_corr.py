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

base_dir = "C:/Users/memes/Documents/SIPS/Correction"

correction_era_3 = np.load('correction_era_3.npz')
correction_era_10 = np.load('correction_era_10.npz')
correction_hrrr_3 = np.load('correction_hrrr_3.npz')
correction_hrrr_10 = np.load('correction_hrrr_10.npz')

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

# Load data
Obs_May_2024 = pd.read_pickle("Obs_NM_2024-05-01_2024-05-31_3m_10m.pkl")
Obs_July_2024 = pd.read_pickle("Obs_NM_2024-07-01_2024-07-31_3m_10m.pkl")
Obs_July_2024 = Obs_July_2024.set_index("timestamp")

Obs_May_July_2024 = pd.concat([Obs_May_2024, Obs_July_2024])

# Pick one station (replace with your actual ID, e.g. 'STATION123')
station_id = "NMC60"

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


def find_closest_HRRR_loc(hrrr: pd.DataFrame, hrrr_coords_to_use):
    """
    hrrr: HRRR dataframe with columns ['lat', 'lon', 'fsr', ...]
    hrrr_coords_to_use: array-like [lat, lon]
    """

    # Compute squared distance between requested point and all HRRR grid points
    dist = (hrrr["longitude"].values - hrrr_coords_to_use[1])**2 + \
           (hrrr["latitude"].values - hrrr_coords_to_use[0])**2

    # Index of closest location
    idx = np.argmin(dist)

    # Extract closest row
    hrrr_loc = hrrr.iloc[idx]

    # Extract lat/lon
    closest_lat = hrrr_loc["latitude"]
    closest_lon = hrrr_loc["longitude"]

    # Extract fsr if exists
    fsr_val = hrrr_loc["fsr"] if "fsr" in hrrr.columns else np.nan

    return hrrr_loc, idx, closest_lat, closest_lon, fsr_val

        
hrrr_loc, idx, closest_lat, closest_lon, fsr_val = find_closest_HRRR_loc(HRRR_May_2024, [latitude_station[0], longitude_station[0]])       

hrrr_index = np.where(hrrr_latitude == closest_lat)[0]
HRRR_May_2024_sorted = HRRR_May_2024.sort_values(by="longitude").reset_index(drop=True)

tol = 1e-3  # tolerance in degrees (~100 m)
HRRR_May_2024_station = HRRR_May_2024_sorted[
    np.isclose(HRRR_May_2024_sorted["latitude"], closest_lat, atol=tol) &
    np.isclose(HRRR_May_2024_sorted["longitude"], closest_lon, atol=tol)
]


plt.figure()
plt.plot(Obs_station_hourly.index,Obs_station_hourly.windspeed,label='obs (3m)',alpha=0.8)
plt.plot(Obs_station_hourly.index[0:742],HRRR_May_2024_station.windspeed10[0:742],label='HRRR (10m)',alpha=0.5)
plt.legend(loc='upper right',fontsize=10)
plt.xlabel('Date')
plt.ylabel('Wind speed (m/s) at NMC60, NM')
#plt.xticks([0,5,10,15,20,30,60,90,120])
plt.legend(loc='upper right',fontsize=10)
plt.show()
#plt.autofmt_xdate()
#plt.tight_layout()
#plt.savefig("obs num time resolution NM.png")
#plt.xlim([datetime.datetime(2024,8,26,0,0), datetime.datetime(2024,9,10,0,0)])
#plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%YYYY-%MM'))

plt.figure()
plt.plot(Obs_station_hourly.index,Obs_station_hourly.windspeed_10m,label='obs',alpha=0.8)
plt.plot(Obs_station_hourly.index,correction_hrrr_10_wspd[hrrr_index[0],0:744],label='HRRR corrected',alpha=0.7)
plt.plot(Obs_station_hourly.index[0:742],HRRR_May_2024_station.windspeed10[0:742],label='HRRR',alpha=0.5)
plt.legend(loc='upper right',fontsize=10)
plt.xlabel('Date')
plt.ylabel('Wind speed (m/s) at 10m height (NMC60)')
#plt.xticks([0,5,10,15,20,30,60,90,120])
plt.legend(loc='upper right',fontsize=10)
plt.show()

plt.figure()
plt.plot(Obs_station_hourly.index,Obs_station_hourly.windspeed_3m,label='obs',alpha=0.8)
plt.plot(Obs_station_hourly.index,correction_hrrr_3_wspd[hrrr_index[0],0:744],label='HRRR corrected',alpha=0.7)
plt.plot(Obs_station_hourly.index[0:742],HRRR_May_2024_station.windspeed_3m[0:742],label='HRRR',alpha=0.5)
plt.legend(loc='upper right',fontsize=10)
plt.xlabel('Date')
plt.ylabel('Wind speed (m/s) at 3m height (NMC60)')
#plt.xticks([0,5,10,15,20,30,60,90,120])
plt.legend(loc='upper right',fontsize=10)
plt.show()

plt.figure()
plt.plot(Obs_station_hourly.index,Obs_station_hourly.winddirection,label='obs')
plt.plot(Obs_station_hourly.index,correction_hrrr_10_wdir[hrrr_index[0],0:744],label='HRRR corrected')
plt.legend(loc='upper right',fontsize=10)
plt.xlabel('Date')
plt.ylabel('Wind direction (deg) at 10m height (NMC60)')
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
plt.xlabel('Difference in wind speed (m/s) at 3m height (NMC60)')
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
plt.xlabel('Difference in wind speed (m/s) at 10m height (NMC60)')
plt.ylabel('Frequency')
plt.xticks([-14,-12,-10,-8,-6,-4,-2,0,2,4,6,8,10,12,14])
plt.legend(loc='upper right',fontsize=10)
plt.show()


