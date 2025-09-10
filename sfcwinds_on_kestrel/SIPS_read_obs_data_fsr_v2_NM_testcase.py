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

# Read metadata to filter what we want
meta = pd.read_csv("/kfs2/projects/sfcwinds/observations/metadata_CONUS.csv")


# filter which data to read
base_dir = "/kfs2/projects/sfcwinds/observations"
states_SW = ["NM"]
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
time_deltas = timeseries_sorted.diff().dropna()
delta_minutes = time_deltas.dt.total_seconds() / 60  # in minutes

time_start = timeseries_sorted.min()
time_end = timeseries_sorted.max()
dt_start = datetime.strftime(time_start, "%Y-%m-%d %H:%M:%S")
dt_end = datetime.strftime(time_end, "%Y-%m-%d %H:%M:%S")
dtp_start = datetime.strptime(dt_start, "%Y-%m-%d %H:%M:%S")
dtp_end = datetime.strptime(dt_end, "%Y-%m-%d %H:%M:%S")
time_total = dtp_end - dtp_start
time_missing_days = (delta_minutes > 1440).sum()
time_missing_days_indices = timeseries[:-1][np.array(delta_minutes > 1440)]
time_availability_days = 1-(time_missing_days*1440)/(time_total.total_seconds()/60)

time_missing_hours = (delta_minutes > 60).sum()
time_missing_hours_indices = timeseries[:-1][np.array(delta_minutes > 60)]
time_availability_hours = 1-(time_missing_hours)/(time_total.total_seconds()/3600)

with open('/kfs2/projects/sfcwinds/scripts/data_availability_hours.txt', 'w') as f:
    f.write(str(time_availability_hours))

with open('/kfs2/projects/sfcwinds/scripts/time_missing_hours.txt', 'w') as f:
    f.write(str(time_missing_hours))

latitude_stations = obs.groupby('station_id')['lat'].apply(list)
longitude_stations = obs.groupby('station_id')['lon'].apply(list)
station_id = obs['station_id'].unique()   

#plt.figure()
#plt.hist(delta_minutes, bins=60, range=(0, 100), edgecolor='none',alpha=0.5,label='NM')
#plt.legend(loc='upper right',fontsize=10)
#plt.xlabel('Time difference between samples (minutes)')
#plt.ylabel('Frequency')
#plt.xticks([0,5,10,15,20,30,60,90,120])
#plt.legend(loc='upper right',fontsize=10)

# Read surfce roughness from an hourly file  - maybe later use the daily-current file, but surface toughness should not change that often.
#hrrr = xr.open_dataset("HRRR/hrrr_US_SW_2024-12-31.nc")   # can be any file
#hrrr = xr.open_dataset("C:/Users/memes/Documents/SIPS/HRRR US/hrrr_US_2024-12-30.nc")   # can be any file
#hrrr = xr.open_mfdataset('C:/Users/memes/Documents/SIPS/HRRR/*.nc', chunks={'time': 100})   # multiple files

HRRR_folder = "/kfs2/projects/sfcwinds/test_case"
area = "NM"
#date_string = "2024-12-30"

file_pattern = os.path.join(HRRR_folder, f"HRRR_{area}*.nc")
file_list = sorted(glob.glob(file_pattern))
if file_list:
    hrrr = xr.open_mfdataset(file_list[:], engine="netcdf4", concat_dim ="valid_time",combine='nested', chunks="auto", parallel = True)
    
    
# Get z0 at a specific location, and mean over the time steps (should not change with time). The z0 parameter is called "fsr"

hrrr = hrrr.sortby('valid_time')
valid_times_coord = hrrr['valid_time']
valid_times_array = valid_times_coord.values
hrrr['longitude'] = xr.where(hrrr['longitude'] > 180, hrrr['longitude'] - 360, hrrr['longitude'])
hrrr["wspd10"] = (hrrr.u10**2 + hrrr.v10**2)**0.5

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

#station_lat_list = [d['station_lat'] for d in results if 'station_lat' in d]

#index_station1 = station_lat_list[station_lat_list==31.702]


obs['z0'] = obs['station_id'].map(fsr_dict)
obs['windspeed_3m'] = np.where(obs['height'] == 3, obs['windspeed'], obs['windspeed']*(np.log(3/obs['z0']))/(np.log(obs['height']/obs['z0'])))
obs['windspeed_10m'] = np.where(obs['height'] == 10, obs['windspeed'], obs['windspeed']*(np.log(10/obs['z0']))/(np.log(obs['height']/obs['z0'])))

#fsr_dict_fixed = {k: [v] for k, v in fsr_dict.items()}     # Wrap scalar values in lists
#fsr_df = pd.DataFrame(fsr_dict_fixed)                      # Create DataFrame
#fsr_df.to_csv('fsr_NM.csv', index=False)                   # Save the DataFrame to a CSV file, index=False prevents Pandas from writing the DataFrame index as a column in the CSV

#hrrr_loc["windspeed_10m"] = (hrrr_loc.u10**2 + hrrr_loc.v10**2)**0.5
#hrrr_loc['z0'] = [d['fsr'] for d in results if 'fsr' in d]
#hrrr_loc['z0'] = [arr[-1] for arr in fsr_list]
#hrrr_loc['windspeed_3m'] = hrrr_loc['windspeed_10m']*(np.log(3/hrrr_loc['z0']))/(np.log(10/hrrr_loc['z0']))

obs_ws_10m_array = np.array(obs['windspeed_10m'])
obs_ws_3m_array = np.array(obs['windspeed_3m'])
#hrrr_ws_10m_array = np.array(hrrr_loc["windspeed_10m"])
#hrrr_ws_3m_array = np.array(hrrr_loc["windspeed_3m"])

# Compare obs 10m vs HRRR (NM)

import pytz
import datetime
import matplotlib.dates as mdates

#hrrr_loc = hrrr_loc.assign_coords(valid_time=('time', hrrr_loc['valid_time'].to_index().tz_localize('UTC')))
#hrrr_loc = hrrr_loc.assign_coords(valid_time=('time', hrrr_loc['valid_time'].tz_localize('UTC')))
#hrrr_loc = hrrr_loc.assign_coords(valid_time=('time', hrrr_loc.indexes['valid_time'].tz_localize('UTC')))

for stid  in obs.station_id.unique()[:]:
    # Time series of each station
    station_obs = obs[obs.station_id == stid]
    #if len(station_obs.dropna(subset = "windspeed")) > 0:

    # resample to 15 min
    #station_obs = station_obs.resample("15min", on="timestamp").mean(numeric_only=True)

df_hrrr = hrrr_loc.drop_dims("isobaricInhPa").to_dataframe()
df_hrrr.index = df_hrrr.index.tz_localize("UTC")
#df_hrrr.index = df_hrrr.index.tz_localize(None)
df_hrrr['timestamp'] = df_hrrr.index

station_obs2 = station_obs.reset_index()
station_obs2['timestamp'] = pd.to_datetime(station_obs2['timestamp'], utc=True)
station_obs2 = station_obs2.set_index('timestamp')
station_obs2_15min = station_obs2.resample("15T").mean(numeric_only=True)

# Merge on aligned timestamps
merged = pd.merge(
    df_hrrr.reset_index(),
    station_obs2_15min.reset_index(),
    left_on="timestamp",
    right_on="timestamp",
    how="inner"
)


#merged = station_obs.merge(df_hrrr,left_on = "timestamp",right_index = True,how = "inner")

# Narrow time series
#merged_time = 

plt.figure()
plt.plot(merged.valid_time[~np.isnan(merged.windspeed_10m)],merged.windspeed_10m[~np.isnan(merged.windspeed_10m)],label='obs')
plt.plot(merged.valid_time[~np.isnan(merged.windspeed_10m)],merged.wspd10[~np.isnan(merged.windspeed_10m)],label='HRRR')
plt.legend(loc='upper right',fontsize=10)
plt.xlabel('Time')
plt.ylabel('Wind speed (m/s) at 10m height')
#plt.xticks([0,5,10,15,20,30,60,90,120])
plt.legend(loc='upper right',fontsize=10)
plt.show()
#plt.autofmt_xdate()
#plt.tight_layout()
#plt.savefig("obs num time resolution NM.png")
plt.savefig("/kfs2/projects/sfcwinds/scripts/obs_hrrr_10m_windspeed_NM_testcase.png")
#plt.xlim([datetime.datetime(2024,8,26,0,0), datetime.datetime(2024,9,10,0,0)])
#plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))

merged.to_csv("/kfs2/projects/sfcwinds/scripts/merged.csv", index=False) 

