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
import pickle

# Read metadata to filter what we want
meta = pd.read_csv("/kfs2/projects/sfcwinds/observations/metadata_CONUS.csv")


# filter which data to read
base_dir = "/kfs2/projects/sfcwinds/observations"
states_SW = ["CA"]
#states_SW = ["CA","TX","AZ","NM","CO","UT","NV"]
stations = meta[meta.state.isin(states_SW)].station_id.values
networks = ["*"]
years    = ["*"]

#Read observations according to filter

years = [str(y) for y in years]

all_dfs = []


for net in networks:
    for station in stations:
        for year in years:
            pattern = os.path.join(base_dir, net, station, f"{year}.parquet")
            found = glob.glob(pattern)

            for f in found:
                try:
                    df = pd.read_parquet(f)
                    # Extract station_id from the path (e.g., .../CoAgMet/HOT01/2023.parquet)
                    station_id = os.path.basename(os.path.dirname(f))  # 'HOT01'
                    df["station_id"] = station_id
                    all_dfs.append(df)
                except Exception as e:
                    print(f"Failed to read {f}: {e}")



if not all_dfs:
    print("No files found.")
else:
    print(f"{len(all_dfs)} files found.")
    obs = pd.concat(all_dfs, ignore_index=True)

# Merge with metadata
obs = obs.merge(meta[["station_id", "lat", "lon", "height", "elev", "source_network", "state"]], on="station_id", how="left")

time = obs.timestamp.to_frame()  # Convert to pandas.DatetimeIndex
time['datetime'] = pd.to_datetime(time['timestamp'],utc=True)
timeseries = pd.Series(time['datetime'])
timeseries_sorted = timeseries.sort_values()
time_deltas = timeseries.diff().dropna()
delta_minutes = time_deltas.dt.total_seconds() / 60  # in minutes

time_start = timeseries_sorted.min()
time_end = timeseries_sorted.max()
dt_start = datetime.strftime(time_start, "%Y-%m-%d %H:%M:%S")
dt_end = datetime.strftime(time_end, "%Y-%m-%d %H:%M:%S")
dtp_start = datetime.strptime(dt_start, "%Y-%m-%d %H:%M:%S")
dtp_end = datetime.strptime(dt_end, "%Y-%m-%d %H:%M:%S")
time_total = dtp_end - dtp_start
time_missing = (delta_minutes > 1440).sum()
time_missing_indices = timeseries[:-1][np.array(delta_minutes > 1440)]
time_availability = 1-(time_missing*1440)/(time_total.total_seconds()/60)

latitude_stations = obs.groupby('station_id')['lat'].apply(list)
longitude_stations = obs.groupby('station_id')['lon'].apply(list)

latitude_obs = np.array([row[0] for row in latitude_stations])
longitude_obs = np.array([row[0] for row in longitude_stations])+360


with open("/kfs2/projects/sfcwinds/scripts/dt_start.txt", "w") as f:
    f.write(dt_start)

with open("/kfs2/projects/sfcwinds/scripts/dt_end.txt", "w") as f:
    f.write(dt_end)

with open('/kfs2/projects/sfcwinds/scripts/data_availability.txt', 'w') as f:
    f.write(str(time_availability))

with open('/kfs2/projects/sfcwinds/scripts/time_missing.txt', 'w') as f:
    f.write(str(time_missing))


# Convert the Timedelta Series to string representation
timeseries_str = timeseries.astype(str)
time_missing_series_str = time_missing_indices.astype(str)

# Save to CSV
timeseries_str.to_csv("/kfs2/projects/sfcwinds/scripts/timeseries.csv", index=False)
time_missing_series_str.to_csv("/kfs2/projects/sfcwinds/scripts/time_missing_indices.csv", index=False)

np.savetxt('/kfs2/projects/sfcwinds/scripts/latitude_obs.txt', latitude_obs, delimiter=',')    
np.savetxt('/kfs2/projects/sfcwinds/scripts/longitude_obs.txt', longitude_obs, delimiter=',')    

    
# Plot histograms (norm)

plt.figure()
#plt.title(obs.station + f", {ds.time.dt.year.values[0]}")
plt.hist(delta_minutes, bins=80, range=(0, 120), edgecolor='none',weights=np.ones(len(delta_minutes))/len(delta_minutes), density=False,alpha=0.5,label='CA')
plt.legend(loc='upper right',fontsize=10)
plt.xlabel('Time difference between samples (minutes)')
plt.ylabel('Frequency')
plt.xticks([0,5,10,15,20,30,60,90,120])
plt.legend(loc='upper right',fontsize=10)
plt.savefig("/kfs2/projects/sfcwinds/scripts/obs time resolution CA.png")
plt.show()

plt.figure()
plt.hist(obs.height, bins=50, range=(0, 10), edgecolor='none',weights=np.ones(len(obs.height))/len(obs.height), density=False,alpha=0.5,label='CA')
plt.xlabel('Height (m)')
plt.ylabel('Frequency')
plt.xticks([0,1,2,3,4,5,6,7,8,9,10])
plt.legend(loc='upper right',fontsize=10)
plt.savefig("/kfs2/projects/sfcwinds/scripts/obs height CA.png")
plt.show()

plt.figure()
plt.hist(obs.windspeed, bins=np.arange(0, 20, 0.5), edgecolor='none',weights=np.ones(len(obs.windspeed))/len(obs.windspeed), density=False,alpha=0.5,label='CA')
plt.xlabel('Wind speed (m/s)')
plt.ylabel('Frequency')
plt.xticks([0,2,4,6,8,10,12,14,16,18,20])
plt.legend(loc='upper right',fontsize=10)
plt.savefig("/kfs2/projects/sfcwinds/scripts/obs windspeed CA.png")
plt.show()

plt.figure()
plt.hist(obs.windspeed, bins=np.arange(0, 20, 0.5), edgecolor='none',weights=np.ones(len(obs.windspeed))/len(obs.windspeed), density=False,alpha=0.5,label='CA mean')
plt.hist(obs.gust, bins=np.arange(0, 20, 0.5), edgecolor='none',weights=np.ones(len(obs.gust))/len(obs.gust), density=False,alpha=0.5,label='CA gust')
plt.xlabel('Wind speed (m/s)')
plt.ylabel('Frequency')
plt.xticks([0,2,4,6,8,10,12,14,16,18,20])
plt.legend(loc='upper right',fontsize=10)
plt.savefig("/kfs2/projects/sfcwinds/scripts/obs windgustspeed CA.png")
plt.show()

plt.figure()
hist3 = plt.hist(obs.winddirection, bins=np.arange(0, 360, 30), edgecolor='none',weights=np.ones(len(obs.winddirection))/len(obs.winddirection), density=False,alpha=0.5,label='CA')
plt.xlabel('Wind direction (deg)')
plt.ylabel('Frequency')
#plt.title('CA')
plt.xticks([0,90,180,270,360])
plt.legend(loc='upper right',fontsize=10)
plt.savefig("/kfs2/projects/sfcwinds/scripts/obs winddirection CA.png")
plt.show()

plt.figure()
plt.hist(obs.gust, bins=np.arange(0, 20, 0.5), edgecolor='none',weights=np.ones(len(obs.gust))/len(obs.gust), density=False,alpha=0.5,label='CA')
plt.xlabel('Gust wind speed (m/s)')
plt.ylabel('Frequency')
#plt.title('CA')
plt.xticks([0,2,4,6,8,10,12,14,16,18,20])
plt.legend(loc='upper right',fontsize=10)
plt.savefig("/kfs2/projects/sfcwinds/scripts/obs gust CA.png")
plt.show()

