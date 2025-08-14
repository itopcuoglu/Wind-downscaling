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
states_SW = ["NM","AZ","NV","CA"]
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
station_id = obs['station_id'].unique()

latitude_obs = np.array([row[0] for row in latitude_stations])
longitude_obs = np.array([row[0] for row in longitude_stations])+360

# Read surfce roughness from an hourly file  - maybe later use the daily-current file, but surface toughness should not change that often.
#hrrr = xr.open_dataset("HRRR/hrrr_US_SW_2024-12-31.nc")   # can be any file
hrrr = xr.open_dataset("/kfs2/projects/sfcwinds/HRRR/hrrr_US_2024-12-30.nc")   # can be any file

# Get z0 at a specific location, and mean over the time steps (should not change with time). The z0 parameter is called "fsr"

hrrr['longitude'] = xr.where(hrrr['longitude'] > 180, hrrr['longitude'] - 360, hrrr['longitude'])

hrrr = hrrr.sortby('valid_time')

hrrr_latitude = np.array(hrrr.latitude)    
hrrr_longitude = np.array(hrrr.longitude)  

latitude_obs = np.array([row[0] for row in latitude_stations])
longitude_obs = np.array([row[0] for row in longitude_stations])
coordinates_obs = list(zip(latitude_obs, longitude_obs))


def find_closest_HRRR_loc(hrrr, hrrr_coords_to_use):
    
    """
    hrrr: HRRR xarray
    hrrr_coords_to_use: array -> [lat, lon]
    
    
    
    """

    dist = (hrrr['longitude'].values - hrrr_coords_to_use[1])**2 + (hrrr['latitude'].values - hrrr_coords_to_use[0])**2
    idx = np.unravel_index(np.argmin(dist), dist.shape)
    closest_lat = hrrr.latitude.values[idx]
    closest_lon = hrrr.longitude.values[idx]

    hrrr_loc =hrrr.sel(x=idx[1], y=idx[0])

    fsr_val = hrrr_loc['fsr'].values if 'fsr' in hrrr_loc else np.nan
    
    return hrrr_loc, idx, closest_lat, closest_lon, fsr_val


def find_nearest_obs(obs, target_lat = latitude_obs, target_lon = longitude_obs ):

    # Compute distance
    obs['distance_to_sandia'] = np.sqrt(
        (obs['lat'] - target_lat)**2 + (obs['lon'] - target_lon)**2
    )
    
    # Find the row with the smallest distance
    closest_station = np.unique(obs[obs.distance == np.nanmin(obs['distance'])].stid)[0]
    
    # Filter data for the closest station
    ts = obs[obs.stid == closest_station]

    return ts


results = []

for lat, lon in zip(latitude_obs, longitude_obs):
    hrrr_loc, idx, grid_lat, grid_lon, fsr_val = find_closest_HRRR_loc(hrrr, [lat, lon])
    results.append({
        'station_lat': lat,
        'station_lon': lon,
        'grid_lat': grid_lat,
        'grid_lon': grid_lon,
        'fsr': fsr_val
    })

fsr_list = [d['fsr'] for d in results if 'fsr' in d]
fsr_final = [arr[-1] for arr in fsr_list]
fsr_final_array = np.array(fsr_final)
fsr_dict = dict(zip(station_id, fsr_final))

fsr_dict_fixed = {k: [v] for k, v in fsr_dict.items()}     # Wrap scalar values in lists
fsr_df = pd.DataFrame(fsr_dict_fixed)                      # Create DataFrame
fsr_df.to_csv('/kfs2/projects/sfcwinds/scripts/fsr_SW.csv', index=False)                   # Save the DataFrame to a CSV file, index=False prevents Pandas from writing the DataFrame index as a column in the CSV

obs['z0'] = obs['station_id'].map(fsr_dict)
obs['windspeed_3m'] = np.where(obs['height'] == 3, obs['windspeed'], obs['windspeed']*(np.log(3/obs['z0']))/(np.log(obs['height']/obs['z0'])))
obs['windspeed_10m'] = np.where(obs['height'] == 10, obs['windspeed'], obs['windspeed']*(np.log(10/obs['z0']))/(np.log(obs['height']/obs['z0'])))


with open("/kfs2/projects/sfcwinds/scripts/dt_start.txt", "w") as f:
    f.write(dt_start)

with open("/kfs2/projects/sfcwinds/scripts/dt_end.txt", "w") as f:
    f.write(dt_end)

with open('/kfs2/projects/sfcwinds/scripts/data_availability.txt', 'w') as f:
    f.write(str(time_availability))

with open('/kfs2/projects/sfcwinds/scripts/time_missing.txt', 'w') as f:
    f.write(str(time_missing))


# Convert the Timedelta Series to string representation
timeseries_str = timeseries_sorted.astype(str)
time_missing_series_str = time_missing_indices.astype(str)

# Save to CSV
#timeseries_str.to_csv("/kfs2/projects/sfcwinds/scripts/timeseries.csv", index=False)
#time_missing_series_str.to_csv("/kfs2/projects/sfcwinds/scripts/time_missing_indices.csv", index=False)

np.savetxt('/kfs2/projects/sfcwinds/scripts/latitude_obs.txt', latitude_obs, delimiter=',')    
np.savetxt('/kfs2/projects/sfcwinds/scripts/longitude_obs.txt', longitude_obs, delimiter=',')    

    
# Plot histograms (norm)

plt.figure()
#plt.title(obs.station + f", {ds.time.dt.year.values[0]}")
plt.hist(delta_minutes, bins=80, range=(0, 120), edgecolor='none',weights=np.ones(len(delta_minutes))/len(delta_minutes), density=False,alpha=0.5,label='SW')
plt.legend(loc='upper right',fontsize=10)
plt.xlabel('Time difference between samples (minutes)')
plt.ylabel('Frequency')
plt.xticks([0,5,10,15,20,30,60,90,120])
plt.legend(loc='upper right',fontsize=10)
plt.savefig("/kfs2/projects/sfcwinds/scripts/obs time resolution SW.png")
plt.show()

plt.figure()
plt.hist(obs.height, bins=50, range=(0, 10), edgecolor='none',weights=np.ones(len(obs.height))/len(obs.height), density=False,alpha=0.5,label='SW')
plt.xlabel('Height (m)')
plt.ylabel('Frequency')
plt.xticks([0,1,2,3,4,5,6,7,8,9,10])
plt.legend(loc='upper right',fontsize=10)
plt.savefig("/kfs2/projects/sfcwinds/scripts/obs height SW.png")
plt.show()

plt.figure()
plt.hist(obs.windspeed, bins=np.arange(0, 20, 0.5), edgecolor='none',weights=np.ones(len(obs.windspeed))/len(obs.windspeed), density=False,alpha=0.5,label='SW')
plt.xlabel('Wind speed (m/s)')
plt.ylabel('Frequency')
plt.xticks([0,2,4,6,8,10,12,14,16,18,20])
plt.legend(loc='upper right',fontsize=10)
plt.savefig("/kfs2/projects/sfcwinds/scripts/obs windspeed SW.png")
plt.show()

plt.figure()
plt.hist(obs.windspeed, bins=np.arange(0, 20, 0.5), edgecolor='none',weights=np.ones(len(obs.windspeed))/len(obs.windspeed), density=False,alpha=0.5,label='SW mean')
plt.hist(obs.gust, bins=np.arange(0, 20, 0.5), edgecolor='none',weights=np.ones(len(obs.gust))/len(obs.gust), density=False,alpha=0.5,label='SW gust')
plt.xlabel('Wind speed (m/s)')
plt.ylabel('Frequency')
plt.xticks([0,2,4,6,8,10,12,14,16,18,20])
plt.legend(loc='upper right',fontsize=10)
plt.savefig("/kfs2/projects/sfcwinds/scripts/obs windgustspeed SW.png")
plt.show()

plt.figure()
hist3 = plt.hist(obs.winddirection, bins=np.arange(0, 360, 30), edgecolor='none',weights=np.ones(len(obs.winddirection))/len(obs.winddirection), density=False,alpha=0.5,label='SW')
plt.xlabel('Wind direction (deg)')
plt.ylabel('Frequency')
#plt.title('CA')
plt.xticks([0,90,180,270,360])
plt.legend(loc='upper right',fontsize=10)
plt.savefig("/kfs2/projects/sfcwinds/scripts/obs winddirection SW.png")
plt.show()

plt.figure()
plt.hist(obs.gust, bins=np.arange(0, 20, 0.5), edgecolor='none',weights=np.ones(len(obs.gust))/len(obs.gust), density=False,alpha=0.5,label='SW')
plt.xlabel('Gust wind speed (m/s)')
plt.ylabel('Frequency')
#plt.title('CA')
plt.xticks([0,2,4,6,8,10,12,14,16,18,20])
plt.legend(loc='upper right',fontsize=10)
plt.savefig("/kfs2/projects/sfcwinds/scripts/obs gust SW.png")
plt.show()

plt.figure()
plt.hist(obs.windspeed_3m, bins=np.arange(0, 20, 0.5), edgecolor='none',weights=np.ones(len(obs.windspeed_3m))/len(obs.windspeed_3m), density=False,alpha=0.5,label='3m SW')
plt.hist(obs.windspeed_10m, bins=np.arange(0, 20, 0.5), edgecolor='none',weights=np.ones(len(obs.windspeed_10m))/len(obs.windspeed_10m), density=False,alpha=0.5,label='10m SW')
plt.xlabel('Wind speed (m/s)')
plt.ylabel('Frequency')
plt.xticks([0,2,4,6,8,10,12,14,16,18,20])
plt.legend(loc='upper right',fontsize=10)
plt.savefig("/kfs2/projects/sfcwinds/scripts/obs 3m 10m windspeed SW.png")
plt.show()

