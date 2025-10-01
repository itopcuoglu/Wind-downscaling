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

# Read metadata to filter what we want
# meta = pd.read_csv("C:/Users/memes/Documents/SIPS/metadata_CONUS.csv")
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

# Read surfce roughness from an hourly file  - maybe later use the daily-current file, but surface toughness should not change that often.
hrrr = xr.open_dataset("/kfs2/projects/sfcwinds/test_case/HRRR_NM_2024-05-01_2024-05-31.nc")   # can be any file
#hrrr = xr.open_dataset("C:/Users/memes/Documents/SIPS/test_case/HRRR_NM_2024-05-01_2024-05-31.nc")   # can be any file
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




#%% Test case by station
for stid in obs["station_id"].unique()[:1]:
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
    

    save = 0
    if save == 1:
        merged.to_csv(f"{stid}"'_3m_hrrr_10m_merged.csv', index=False)
        quantile_10m_data.to_csv(f"{stid}"'_3m_hrrr_10m_quantile_data.csv', index=False)
        quantile_10m_stats.to_csv(f"{stid}"'_3m_hrrr_10m_quantile_stats.csv', index=False)
        
