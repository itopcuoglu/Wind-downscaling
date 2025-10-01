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
meta = pd.read_csv("C:/Users/memes/Documents/SIPS/metadata_CONUS.csv")


# filter which data to read
base_dir = "C:/Users/memes/Documents/SIPS"
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

latitude_stations = obs.groupby('station_id')['lat'].apply(list)
longitude_stations = obs.groupby('station_id')['lon'].apply(list)
station_id = obs['station_id'].unique()   

plt.figure()
plt.hist(delta_minutes, bins=60, range=(0, 100), edgecolor='none',alpha=0.5,label='NM')
plt.legend(loc='upper right',fontsize=10)
plt.xlabel('Time difference between samples (minutes)')
plt.ylabel('Frequency')
#plt.xticks([0,5,10,15,20,30,60,90,120])
plt.legend(loc='upper right',fontsize=10)

# Read surfce roughness from an hourly file  - Julybe later use the daily-current file, but surface toughness should not change that often.
#hrrr = xr.open_dataset("HRRR/hrrr_US_SW_2024-12-31.nc")   # can be any file
hrrr = xr.open_dataset("C:/Users/memes/Documents/SIPS/test_case/HRRR_NM_2024-05-01_2024-05-31.nc")   # can be any file
#hrrr = xr.open_mfdataset('C:/Users/memes/Documents/SIPS/HRRR/*.nc', chunks={'time': 100})   # multiple files

#HRRR_folder = "C:/Users/memes/Documents/SIPS/HRRR"
#area = "US_SW"
#date_string = "2024-12-30"


#file_pattern = os.path.join(HRRR_folder, f"hrrr_{area}*.nc")
#file_list = sorted(glob.glob(file_pattern))
#if file_list:
#    hrrr = xr.open_mfdataset(file_list[:], concat_dim ="valid_time",combine='nested', chunks="auto", parallel = True)
    
# Get z0 at a specific location, and mean over the time steps (should not change with time). The z0 parameter is called "fsr"

hrrr = hrrr.sortby('valid_time')
valid_times_coord = hrrr['valid_time']
valid_times_array = valid_times_coord.values
hrrr['longitude'] = xr.where(hrrr['longitude'] > 180, hrrr['longitude'] - 360, hrrr['longitude'])
hrrr["wspd10"] = (hrrr.u10**2 + hrrr.v10**2)**0.5

hrrr_latitude = np.array(hrrr.latitude)    
hrrr_longitude = np.array(hrrr.longitude)  

def rotate_to_true_north(u10, v10, lon):
    """
    Rotate HRRR 10m wind components to true north orientation.

    Parameters:
    - u10, v10: xarray.DataArrays of shape (time, step)
    - lon: longitude (scalar, degrees east)
    - rotcon_p: rotation constant (default: sin(lat_tan), 38.5° for HRRR)
    - lon_xx_p: reference meridian (default: -97.5°)

    Returns:
    - un10, vn10: xarray.DataArrays of rotated wind components relative to True North
    """
    
    lon = ((lon + 180) % 360) - 180  # make between -180 and 180
                             

    rotcon_p = np.sin(np.radians(38.5))  # for Lambert Conformal (HRRR), 0.6225
    lon_xx_p = -97.5  # reference meridian

    angle2 = rotcon_p * (lon - lon_xx_p) * np.pi / 180
    sinx2 = np.sin(angle2)
    cosx2 = np.cos(angle2)

    # Apply rotation
    un10 = cosx2 * u10 + sinx2 * v10
    vn10 = -sinx2 * u10 + cosx2 * v10

    return un10, vn10

un10, vn10 = rotate_to_true_north(hrrr["u10"], hrrr["v10"], hrrr["longitude"])
hrrr = hrrr.assign(un10=un10, vn10=vn10)

#hrrr.to_netcdf("hrrr.nc")

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

    hrrr_loc =hrrr.isel(y=idx[0], x=idx[1])
    
    fsr_val = hrrr_loc['fsr'].values if 'fsr' in hrrr_loc else np.nan

    return hrrr_loc, closest_lat, closest_lon, idx, fsr_val
 
    
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

station_lat_list = [d['station_lat'] for d in results if 'station_lat' in d]

index_station1 = station_lat_list.index(34.4255)

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

station_id_list = obs.station_id.unique()[:]

import pytz
import datetime
import matplotlib.dates as mdates

#hrrr_loc = hrrr_loc.assign_coords(valid_time=('time', hrrr_loc['valid_time'].to_index().tz_localize('UTC')))
#hrrr_loc = hrrr_loc.assign_coords(valid_time=('time', hrrr_loc['valid_time'].tz_localize('UTC')))
#hrrr_loc = hrrr_loc.assign_coords(valid_time=('time', hrrr_loc.indexes['valid_time'].tz_localize('UTC')))

#for stid  in obs.station_id.unique()[:]:
    # Time series of each station
#    station_obs = obs[obs.station_id == stid]
    #if len(station_obs.dropna(subset = "windspeed")) > 0:        
#        # resample to 15 min
#        #station_obs = station_obs.resample("15min", on="timestamp").mean(numeric_only=True)
#    np.save(f"{stid}.npy", station_obs.values)



#%% Combine quantile stats and contour plots

#quantile_10m_stats.to_csv(f"{stid}"'_3m_hrrr_10m_quantile_stats.csv', index=False)

import glob
import pandas as pd
import os

# Path to your folder (adjust as needed)
folder = "C:/Users/memes/Documents/SIPS/NM obs/Quantile results (May 2024)"

# Read all CSVs into a list of DataFrames
dfs = [pd.read_csv(file) for file in glob.glob(f"{folder}/*_3m_hrrr_10m_quantile_stats.csv")]

# Combine into one DataFrame
combined_df = pd.concat(dfs, ignore_index=True)


for file in glob.glob(f"{folder}/*_3m_hrrr_10m_quantile_stats.csv"):
    df = pd.read_csv(file)
    
    # Extract station ID (text before first "_")
    filename = os.path.basename(file)
    station_id = filename.split("_")[0]
    
    df["station_id"] = station_id
    dfs.append(df)

combined_df = pd.concat(dfs, ignore_index=True)


combined_df_pivot = combined_df.pivot_table(
    index="station_id",
    columns="dataset",
    values=combined_df.columns.drop(["dataset", "station_id"])
)

diffs = {}

for stat in combined_df_pivot.columns.levels[0]:  # e.g. mean, q05, q10, std...
    # wspd
    diffs[(stat, "diff_wspd")] = (
        combined_df_pivot[(stat, "obs_wspd")] - combined_df_pivot[(stat, "hrrr_wspd")]
    )
    # wdir (uncorrected)
    diffs[(stat, "diff_wdir")] = (
        combined_df_pivot[(stat, "obs_wdir")] - combined_df_pivot[(stat, "hrrr_wdir")]
    )
    # wdir vs corrected wdir
    diffs[(stat, "diff_wdir_corr")] = (
        combined_df_pivot[(stat, "obs_wdir")] - combined_df_pivot[(stat, "hrrr_corr_wdir")]
    )

# build a DataFrame with same MultiIndex structure
diff_df = pd.DataFrame(diffs, index=combined_df_pivot.index).dropna()

# flatten the columns: (stat, variable) -> f"{variable}_{stat}"
diff_df.columns = [f"{var}_{stat}" for stat, var in diff_df.columns]

# add station_id and dataset label
diff_df = diff_df.reset_index()
diff_df["dataset"] = "diff"

stations_df = pd.DataFrame({
    "station_id": station_id_list,
    "latitude_obs": latitude_obs,
    "longitude_obs": longitude_obs
}).drop_duplicates()

diff_df_stations = diff_df.merge(stations_df, on="station_id", how="left")



plt.figure()
hist1 = plt.hist(diff_df.diff_wspd_mean, bins=np.arange(-5, 5, 0.2), edgecolor='none',weights=np.ones(len(diff_df.diff_wspd_mean))/len(diff_df.diff_wspd_mean), density=False,alpha=0.5,label= 'Mean')
hist1 = plt.hist(diff_df.diff_wspd_std, bins=np.arange(-5, 5, 0.2), edgecolor='none',weights=np.ones(len(diff_df.diff_wspd_std))/len(diff_df.diff_wspd_std), density=False,alpha=0.5,label= 'Std')
plt.xlabel("Difference in wind speed (m/s) NM obs 3m - hrrr 10m")
plt.ylabel("Frequency")
plt.xticks([-5,-4,-3,-2,-1,0,1,2,3,4,5])
plt.legend(loc='upper right',fontsize=10)
plt.savefig("NM 3m vs hrrr 10m diff wind speed histogram test case NM May 2024.png")
plt.show()

plt.figure()
hist1 = plt.hist(diff_df.diff_wdir_mean, bins=np.arange(-100, 100, 3), edgecolor='none',weights=np.ones(len(diff_df.diff_wdir_mean))/len(diff_df.diff_wdir_mean), density=False,alpha=0.5,label= 'Mean')
hist2 = plt.hist(diff_df.diff_wdir_std, bins=np.arange(-100, 100, 3), edgecolor='none',weights=np.ones(len(diff_df.diff_wdir_std))/len(diff_df.diff_wdir_std), density=False,alpha=0.5,label='Std')
plt.xlabel("Difference in wind direction (deg) NM obs - hrrr")
plt.ylabel("Frequency")
plt.legend(loc='upper right',fontsize=10)
plt.savefig("NM vs hrrr 10m wind direction histogram test case NM May 2024.png")
plt.show()

plt.figure()
hist1 = plt.hist(diff_df.diff_wdir_corr_mean, bins=np.arange(-100, 100, 3), edgecolor='none',weights=np.ones(len(diff_df.diff_wdir_corr_mean))/len(diff_df.diff_wdir_corr_mean), density=False,alpha=0.5,label= 'Mean')
hist2 = plt.hist(diff_df.diff_wdir_corr_std, bins=np.arange(-100, 100, 3), edgecolor='none',weights=np.ones(len(diff_df.diff_wdir_corr_std))/len(diff_df.diff_wdir_corr_std), density=False,alpha=0.5,label='Std')
plt.xlabel("Difference in wind direction (deg) NM obs - hrrr corr")
plt.ylabel("Frequency")
plt.legend(loc='upper right',fontsize=10)
plt.savefig("NM vs hrrr corr 10m wind direction histogram test case NM May 2024.png")
plt.show()




import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import rasterio
from rasterio.plot import show
import os
import pandas as pd
import xarray as xr
import glob
import numpy as np
import sqlite3
import plotly.express as px
from geopy.distance import geodesic
import matplotlib.colors as mcolors

#combined_df["windspeed_diff"] = combined_df.groupby("station_id")["windspeed"].diff()

#hrrr_plot = hrrr.sel(valid_time=pd.to_datetime(date_string), method = "nearest")

# Create a figure with a map projection
fig, ax = plt.subplots(figsize=(10, 6), subplot_kw={'projection': ccrs.PlateCarree()})
# plt.title(f"HRRR day: {hrrr_plot.time.values} \n Metar day: {metar_time}")

# Add map features
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.add_feature(cfeature.STATES, linestyle='-', linewidth=0.5, edgecolor="black") 
ax.add_feature(cfeature.LAND, edgecolor='black', alpha=0.3)
ax.add_feature(cfeature.LAKES, edgecolor='black', alpha=0.3)
ax.add_feature(cfeature.RIVERS, edgecolor='blue', alpha=0.3)

gl = ax.gridlines(draw_labels=True, linestyle="-", linewidth=0.5, alpha=0.0)
gl.right_labels = False  # Hide right labels
gl.top_labels = False  # Hide top labels

ax.set_extent([-110,-102, 31.5, 37.5], crs=ccrs.PlateCarree())  

ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

# HRRR data
#sc = contour = plt.pcolormesh(diff_df_stations.latitude_obs, 
                         #diff_df_stations.longitude_obs, 
                         #diff_df_stations.diff_wspd_mean.squeeze(), cmap="gnuplot", shading="auto", transform=ccrs.PlateCarree())

# Scatter plot with color scale
#sc = ax.scatter(diff_df_stations.longitude_obs,diff_df_stations.latitude_obs, c=diff_df_stations.diff_wspd_mean, cmap='rainbow', s=20)
#sc = ax.scatter(diff_df_stations.longitude_obs,diff_df_stations.latitude_obs, c=diff_df_stations.diff_wdir_mean, cmap='rainbow', s=20)
sc = ax.scatter(diff_df_stations.longitude_obs,diff_df_stations.latitude_obs, c=diff_df_stations.diff_wdir_corr_mean, cmap='rainbow', s=20)
#ax.scatter(diff_df_stations.longitude_obs[14],diff_df_stations.latitude_obs[14], color='k', s=50)


# Add a colorbar to show the mapping
#cbar = plt.colorbar(sc,ax=ax,orientation='vertical', label="Difference in wind speed (m/s) obs - hrrr")
#cbar = plt.colorbar(sc,ax=ax,orientation='vertical', label="Difference in wind direction (deg) obs - hrrr")
cbar = plt.colorbar(sc,ax=ax,orientation='vertical', label="Difference in wind direction (deg) obs - hrrr corr")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")



#%% Test case by station
for stid in obs["station_id"].unique():
    station_obs_test2 = obs[obs["station_id"] == stid]
    #station_obs_test2 = obs[obs["station_id"] == "CZZN5"]       # single station

#station_obs_test = np.load('ACON5.npy', allow_pickle=True)   # load array
#lat = station_obs_test[0,7]
#lon = station_obs_test[0,8]

    hrrr_loc, idx, grid_lat, grid_lon, fsr_val = find_closest_HRRR_loc(hrrr, [station_obs_test2.lat.iloc[0], station_obs_test2.lon.iloc[0]])
    
    df_hrrr = hrrr_loc.drop_dims("isobaricInhPa").to_dataframe()
    #df_hrrr.index = df_hrrr.index.tz_localize(None)
    df_hrrr.index = df_hrrr.index.tz_localize("UTC")
    df_hrrr['timestamp'] = df_hrrr.index
    
    #df_hrrr.to_pickle('df_hrrr.pkl')
    
    #hrrr_df = hrrr_df[["u10", "v10", "wspd10"]].to_dataframe().reset_index()
    
    # Ensure timestamps are timezone-aware UTC
    #obs['timestamp'] = pd.to_datetime(obs['timestamp'], utc=True)
    #obs = obs.set_index('timestamp')
    
    # Resample obs to 15-minute mean (same as HRRR interval)
    #obs_15min = obs.resample("15T").mean(numeric_only=True)
    
    station_obs2 = station_obs_test2.reset_index()
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
    
    #merged = obs.merge(df_hrrr,left_on = "timestamp",right_index = True,how = "inner")
    
    #merged = pd.concat([obs, df_hrrr], axis=1, join="inner")
    
    # Narrow time series
    #merged_time = 
    
    merged["wdir10"] = np.degrees(np.arctan2(merged["u10"], merged["v10"])) +180
    merged["wdir10_corr"] = np.degrees(np.arctan2(merged["un10"], merged["vn10"])) +180
    stid = station_obs2["station_id"].unique()[0]
    height = station_obs2["height"].unique()[0]
    
    #merged.to_csv(f"{stid}'_hrrr_merged_test_case.csv', index=False)
    
    fig, ax = plt.subplots()
    ax.plot(merged.valid_time[~np.isnan(merged.windspeed_3m)],merged.windspeed_3m[~np.isnan(merged.windspeed_3m)],label= f"{stid}"' 3m')
    ax.plot(merged.valid_time[~np.isnan(merged.windspeed_3m)],merged.wspd10[~np.isnan(merged.windspeed_3m)],label='HRRR 10m')
    ax.legend(loc='upper right',fontsize=10)
    ax.set_xlabel('Date')
    ax.set_ylabel('Wind speed (m/s)')
    ax.legend(loc='upper right',fontsize=10)
    fig.autofmt_xdate()
    fig.savefig(f"{stid} 3m vs hrrr 10m wind speed test case NM May 2024.png")
    #plt.tight_layout()
    #plt.savefig("obs num time resolution NM.png")
    #plt.xlim([datetime.datetime(2024,8,26,0,0), datetime.datetime(2024,9,10,0,0)])
    #plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))
    
    fig, ax = plt.subplots()
    ax.plot(merged.valid_time[~np.isnan(merged.winddirection)],merged.winddirection[~np.isnan(merged.winddirection)],label= f"{stid}"' 3m')
    ax.plot(merged.valid_time[~np.isnan(merged.winddirection)],merged.wdir10[~np.isnan(merged.winddirection)],label='HRRR 10m')
    ax.legend(loc='upper right',fontsize=10)
    ax.set_xlabel('Date')
    ax.set_ylabel('Wind direction (deg)')
    ax.legend(loc='upper right',fontsize=10)
    fig.autofmt_xdate()
    fig.savefig(f"{stid} 3m  hrrr 10m wind direction test case NM May 2024.png")
    
    # Quantile mapping
    
    import numpy as np
    from scipy.stats import norm, gamma, erlang, expon
    
    # Data
    u1, s1 = 100,20
    bias = 20
    bias_d = 2
    n = 100
    q_ = 20             
    dist1 = norm(loc=u1, scale=s1)
    dist2 = norm(loc=u1+bias_d, scale=s1)
    
    quantiles = [round(x*0.05,2) for x in range(1,q_)]      # for 5% increments up to 95%
    
    q_dist1 = [dist1.ppf(x*0.05) for x in range(1,q_)]
    q_dist2 = [dist2.ppf(x*0.05) for x in range(1,q_)]
    
    # Distribution real sample
    ref_dataset = np.random.normal(u1,s1,n).round(2)
    # Sub-estimated
    model_present = (q_dist1 - np.random.normal(bias,2,q_-1)).round(2)
    # Future model with the same bias
    model_future = (q_dist2 - np.random.normal(bias,2,q_-1)).round(2)
    
    # Quantile mapping obs vs hrrr (wind speed)
    obs_windspeed_3m = merged.windspeed_3m[~np.isnan(merged.windspeed_3m)]
    hrrr_windspeed_10m = merged.wspd10[~np.isnan(merged.windspeed_3m)]
    
    dist_obs_ws = norm(loc=obs_windspeed_3m.mean(), scale=obs_windspeed_3m.std())
    dist_hrrr_ws = norm(loc=hrrr_windspeed_10m.mean(), scale=hrrr_windspeed_10m.std())
    
    q_dist_obs_ws = [dist_obs_ws.ppf(x*0.05) for x in range(1,q_)]
    q_dist_hrrr_ws = [dist_hrrr_ws.ppf(x*0.05) for x in range(1,q_)]
    q_dist_diff_ws = np.array(q_dist_obs_ws)-np.array(q_dist_hrrr_ws)
    
    plt.figure()
    hist1 = plt.hist(obs_windspeed_3m, bins=np.arange(0, 16, 0.3), edgecolor='none',weights=np.ones(len(obs_windspeed_3m))/len(obs_windspeed_3m), density=False,alpha=0.5,label= f"{stid}"' 3m')
    hist2 = plt.hist(hrrr_windspeed_10m, bins=np.arange(0, 16, 0.3), edgecolor='none',weights=np.ones(len(hrrr_windspeed_10m))/len(hrrr_windspeed_10m), density=False,alpha=0.5,label='hrrr 10m')
    plt.xlabel("Wind speed (m/s)")
    plt.ylabel("Frequency")
    plt.legend(loc='upper right',fontsize=10)
    plt.savefig(f"{stid} 3m vs hrrr 10m wind speed histogram test case NM May 2024.png")
    plt.show()
    
    plt.figure()
    plt.plot(quantiles,q_dist_obs_ws,alpha=0.5,label= f"{stid}")
    plt.plot(quantiles,q_dist_hrrr_ws,alpha=0.5,label='hrrr')
    plt.xlabel("Quantiles")
    plt.ylabel("Values")
    plt.xticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
    plt.legend(loc='upper left',fontsize=10)
    plt.savefig(f"{stid} 3m vs hrrr 10m wind speed quantiles test case NM May 2024.png")
    plt.show()
    
    plt.figure()
    plt.plot(quantiles,q_dist_diff_ws,alpha=0.5,label=f"{stid}"'-hrrr')
    plt.xlabel("Quantiles")
    plt.ylabel("Difference")
    plt.xticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
    plt.legend(loc='upper left',fontsize=10)
    plt.savefig(f"{stid} 3m vs hrrr 10m wind speed quantiles difference test case NM May 2024.png")
    plt.show()
    
    
    # Quantile mapping obs vs hrrr (wind speed)
    obs_winddirection_10m = merged.winddirection[~np.isnan(merged.winddirection)]
    hrrr_winddirection_10m = merged.wdir10[~np.isnan(merged.winddirection)]
    hrrr_winddirection_corr_10m = merged.wdir10_corr[~np.isnan(merged.winddirection)]
    
    dist_obs_wdir = norm(loc=obs_winddirection_10m.mean(), scale=obs_winddirection_10m.std())
    dist_hrrr_wdir = norm(loc=hrrr_winddirection_10m.mean(), scale=hrrr_winddirection_10m.std())
    dist_hrrr_corr_wdir = norm(loc=hrrr_winddirection_corr_10m.mean(), scale=hrrr_winddirection_corr_10m.std())
    
    q_dist_obs_wdir = [dist_obs_wdir.ppf(x*0.05) for x in range(1,q_)]
    q_dist_hrrr_wdir = [dist_hrrr_wdir.ppf(x*0.05) for x in range(1,q_)]
    q_dist_diff_wdir = np.array(q_dist_obs_wdir)-np.array(q_dist_hrrr_wdir)
    q_dist_hrrr_corr_wdir = [dist_hrrr_corr_wdir.ppf(x*0.05) for x in range(1,q_)]
    q_dist_diff_corr_wdir = np.array(q_dist_obs_wdir)-np.array(q_dist_hrrr_corr_wdir)
    
    plt.figure()
    hist1 = plt.hist(obs_winddirection_10m, bins=np.arange(0, 360, 2), edgecolor='none',weights=np.ones(len(obs_winddirection_10m))/len(obs_winddirection_10m), density=False,alpha=0.5,label= f"{stid}"' 3m')
    hist2 = plt.hist(hrrr_winddirection_10m, bins=np.arange(0, 360, 2), edgecolor='none',weights=np.ones(len(hrrr_winddirection_10m))/len(hrrr_winddirection_10m), density=False,alpha=0.5,label='hrrr 10m')
    hist2 = plt.hist(hrrr_winddirection_corr_10m, bins=np.arange(0, 360, 2), edgecolor='none',weights=np.ones(len(hrrr_winddirection_corr_10m))/len(hrrr_winddirection_corr_10m), density=False,alpha=0.5,label='hrrr corr 10m')
    plt.xlabel("Wind direction (deg)")
    plt.ylabel("Frequency")
    plt.legend(loc='upper left',fontsize=10)
    plt.savefig(f"{stid} 3m vs hrrr 10m wind direction histogram test case NM May 2024.png")
    plt.show()
    
    plt.figure()
    plt.plot(quantiles,q_dist_obs_wdir,alpha=0.5,label= f"{stid}"' 3m')
    plt.plot(quantiles,q_dist_hrrr_wdir,alpha=0.5,label='hrrr 10m')
    plt.plot(quantiles,q_dist_hrrr_corr_wdir,alpha=0.5,label='hrrr corr 10m')
    plt.xlabel("Quantiles")
    plt.ylabel("Values")
    plt.xticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
    plt.legend(loc='upper left',fontsize=10)
    plt.savefig(f"{stid} 3m vs hrrr 10m wind direction quantiles test case NM May 2024.png")
    plt.show()
    
    plt.figure()
    plt.plot(quantiles,q_dist_diff_wdir,alpha=0.5,label=f"{stid}"'-hrrr')
    plt.plot(quantiles,q_dist_diff_corr_wdir,alpha=0.5,label=f"{stid}"'-hrrr corr')
    plt.xlabel("Quantiles")
    plt.ylabel("Difference")
    plt.xticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
    plt.legend(loc='upper left',fontsize=10)
    plt.savefig(f"{stid} 3m vs hrrr 10m wind direction quantiles difference test case NM May 2024.png")
    plt.show()
    
    
    # Save time series, histogram and quantiles statistics to each station
    
    quantile_10m_data = pd.DataFrame({
        'obs_wspd': obs_windspeed_3m,
        'hrrr_wspd': hrrr_windspeed_10m,
        'obs_wdir': obs_winddirection_10m,
        'hrrr_wdir': hrrr_winddirection_10m,
        'hrrr_corr_wdir': hrrr_winddirection_corr_10m
    })
    
    quantile_10m_stats = pd.DataFrame({
        "dataset": ["obs_wspd", "hrrr_wspd", "obs_wdir", "hrrr_wdir", "hrrr_corr_wdir"],
        "mean": [dist_obs_ws.mean(), dist_hrrr_ws.mean(),
                 dist_obs_wdir.mean(), dist_hrrr_wdir.mean(), dist_hrrr_corr_wdir.mean()],
        "std": [dist_obs_ws.std(), dist_hrrr_ws.std(),
                dist_obs_wdir.std(), dist_hrrr_wdir.std(), dist_hrrr_corr_wdir.std()],
        "q05": [q_dist_obs_ws[0], q_dist_hrrr_ws[0],
                q_dist_obs_wdir[0], q_dist_hrrr_wdir[0], q_dist_hrrr_corr_wdir[0]],
        "q10": [q_dist_obs_ws[1], q_dist_hrrr_ws[1],
                q_dist_obs_wdir[1], q_dist_hrrr_wdir[1], q_dist_hrrr_corr_wdir[1]],
        "q15": [q_dist_obs_ws[2], q_dist_hrrr_ws[2],
                q_dist_obs_wdir[2], q_dist_hrrr_wdir[2], q_dist_hrrr_corr_wdir[2]],
        "q20": [q_dist_obs_ws[3], q_dist_hrrr_ws[3],
                q_dist_obs_wdir[3], q_dist_hrrr_wdir[3], q_dist_hrrr_corr_wdir[3]],
        "q25": [q_dist_obs_ws[4], q_dist_hrrr_ws[4],
                q_dist_obs_wdir[4], q_dist_hrrr_wdir[4], q_dist_hrrr_corr_wdir[4]],
        "q30": [q_dist_obs_ws[5], q_dist_hrrr_ws[5],
                q_dist_obs_wdir[5], q_dist_hrrr_wdir[5], q_dist_hrrr_corr_wdir[5]],
        "q35": [q_dist_obs_ws[6], q_dist_hrrr_ws[6],
                q_dist_obs_wdir[6], q_dist_hrrr_wdir[6], q_dist_hrrr_corr_wdir[6]],
        "q40": [q_dist_obs_ws[7], q_dist_hrrr_ws[7],
                q_dist_obs_wdir[7], q_dist_hrrr_wdir[7], q_dist_hrrr_corr_wdir[7]],
        "q45": [q_dist_obs_ws[8], q_dist_hrrr_ws[8],
                q_dist_obs_wdir[8], q_dist_hrrr_wdir[8], q_dist_hrrr_corr_wdir[8]],
        "q50": [q_dist_obs_ws[9], q_dist_hrrr_ws[9],
                q_dist_obs_wdir[9], q_dist_hrrr_wdir[9], q_dist_hrrr_corr_wdir[9]],
        "q55": [q_dist_obs_ws[10], q_dist_hrrr_ws[10],
                q_dist_obs_wdir[10], q_dist_hrrr_wdir[10], q_dist_hrrr_corr_wdir[10]],
        "q60": [q_dist_obs_ws[11], q_dist_hrrr_ws[11],
                q_dist_obs_wdir[11], q_dist_hrrr_wdir[11], q_dist_hrrr_corr_wdir[11]],
        "q65": [q_dist_obs_ws[12], q_dist_hrrr_ws[12],
                q_dist_obs_wdir[12], q_dist_hrrr_wdir[12], q_dist_hrrr_corr_wdir[12]],
        "q70": [q_dist_obs_ws[13], q_dist_hrrr_ws[13],
                q_dist_obs_wdir[13], q_dist_hrrr_wdir[13], q_dist_hrrr_corr_wdir[13]],
        "q75": [q_dist_obs_ws[14], q_dist_hrrr_ws[14],
                q_dist_obs_wdir[14], q_dist_hrrr_wdir[14], q_dist_hrrr_corr_wdir[14]],
        "q80": [q_dist_obs_ws[15], q_dist_hrrr_ws[15],
                q_dist_obs_wdir[15], q_dist_hrrr_wdir[15], q_dist_hrrr_corr_wdir[15]],
        "q85": [q_dist_obs_ws[16], q_dist_hrrr_ws[16],
                q_dist_obs_wdir[16], q_dist_hrrr_wdir[16], q_dist_hrrr_corr_wdir[16]],
        "q90": [q_dist_obs_ws[17], q_dist_hrrr_ws[17],
                q_dist_obs_wdir[17], q_dist_hrrr_wdir[17], q_dist_hrrr_corr_wdir[17]],
        "q95": [q_dist_obs_ws[18], q_dist_hrrr_ws[18],
                q_dist_obs_wdir[18], q_dist_hrrr_wdir[18], q_dist_hrrr_corr_wdir[18]],
    })
    
    merged.to_csv(f"{stid}"'_3m_hrrr_10m_merged.csv', index=False)
    quantile_10m_data.to_csv(f"{stid}"'_3m_hrrr_10m_quantile_data.csv', index=False)
    quantile_10m_stats.to_csv(f"{stid}"'_3m_hrrr_10m_quantile_stats.csv', index=False)
    
