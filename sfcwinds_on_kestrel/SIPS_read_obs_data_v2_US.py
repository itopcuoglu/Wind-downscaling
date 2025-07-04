import os
import pandas as pd
#import xarray as xr
import glob
#matplotlib widget
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow


# Read metadata to filter what we want
meta = pd.read_csv("/kfs2/projects/sfcwinds/observations/metadata_CONUS.csv")


# filter which data to read
base_dir = "/kfs2/projects/sfcwinds/observations"
states_SW = ["CO","NV","TX","NM","CA","OK","KS","SC","WV","VA","NC","PA","OR","ID","MT","WY","WA","UT","MN","MO","AR","AZ","MS","LA","NH","RI","MI","IL","OH","IA","GA","TN","FL","AL","IN","NY","ME","NE","SD","MA","ND","MD","KY","VT","WI","NJ","DE"]
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

time = obs.timestamp.to_frame()  # CoCOert to pandas.DatetimeIndex
time['datetime'] = pd.to_datetime(time['timestamp'],utc=True)
timeseries = pd.Series(time['datetime'])
time_deltas = timeseries.diff().dropna()
delta_minutes = time_deltas.dt.total_seconds() / 60  # in minCOes


# Plot histograms (num)

plt.figure()
#plt.title(obs.station + f", {ds.time.dt.year.values[0]}")
plt.hist(delta_minutes, bins=80, range=(0, 120), edgecolor='none',alpha=0.5,label='CONUS')
plt.legend(loc='upper right',fontsize=10)
plt.xlabel('Time difference between samples (minutes)')
plt.ylabel('Frequency')
plt.xticks([0,5,10,15,20,30,60,90,120])
plt.legend(loc='upper right',fontsize=10)
plt.savefig("/kfs2/projects/sfcwinds/obs num time resolution CONUS.png")
plt.show()

plt.figure()
plt.hist(obs.height, bins=50, range=(0, 10), edgecolor='none',alpha=0.5,label='CONUS')
plt.xlabel('Height (m)')
plt.ylabel('Frequency')
plt.xticks([0,1,2,3,4,5,6,7,8,9,10])
plt.legend(loc='upper right',fontsize=10)
plt.savefig("/kfs2/projects/sfcwinds/obs num height CONUS.png")
plt.show()

plt.figure()
plt.hist(obs.windspeed, bins=np.arange(0, 20, 0.5), edgecolor='none',alpha=0.5,label='CONUS')
plt.xlabel('Wind speed (m/s)')
plt.ylabel('Frequency')
plt.xticks([0,2,4,6,8,10,12,14,16,18,20])
plt.legend(loc='upper right',fontsize=10)
plt.savefig("/kfs2/projects/sfcwinds/obs num windspeed CONUS.png")
plt.show()

plt.figure()
plt.hist(obs.winddirection, bins=np.arange(0, 360, 30), edgecolor='none',alpha=0.5,label='CONUS')
plt.xlabel('Wind direction (deg)')
plt.ylabel('Frequency')
#plt.title('CA')
plt.xticks([0,90,180,270,360])
plt.legend(loc='upper right',fontsize=10)
plt.savefig("/kfs2/projects/sfcwinds/obs num winddirection CONUS.png")
plt.show()

plt.figure()
plt.hist(obs.gust, bins=np.arange(0, 20, 0.5), edgecolor='none',alpha=0.5,label='CONUS')
plt.xlabel('Gust wind speed (m/s)')
plt.ylabel('Frequency')
#plt.title('CA')
plt.xticks([0,2,4,6,8,10,12,14,16,18,20])
plt.legend(loc='upper right',fontsize=10)
plt.savefig("/kfs2/projects/sfcwinds/obs num gust CONUS.png")
plt.show()

# Plot histograms (norm)

plt.figure()
#plt.title(obs.station + f", {ds.time.dt.year.values[0]}")
plt.hist(delta_minutes, bins=80, range=(0, 120), edgecolor='none',weights=np.ones(len(delta_minutes))/len(delta_minutes), density=False,alpha=0.5,label='CONUS')
plt.legend(loc='upper right',fontsize=10)
plt.xlabel('Time difference between samples (minutes)')
plt.ylabel('Frequency')
plt.xticks([0,5,10,15,20,30,60,90,120])
plt.legend(loc='upper right',fontsize=10)
plt.savefig("/kfs2/projects/sfcwinds/obs time resolution CONUS.png")
plt.show()

plt.figure()
plt.hist(obs.height, bins=50, range=(0, 10), edgecolor='none',weights=np.ones(len(obs.height))/len(obs.height), density=False,alpha=0.5,label='CONUS')
plt.xlabel('Height (m)')
plt.ylabel('Frequency')
plt.xticks([0,1,2,3,4,5,6,7,8,9,10])
plt.legend(loc='upper right',fontsize=10)
plt.savefig("/kfs2/projects/sfcwinds/obs height CONUS.png")
plt.show()

plt.figure()
plt.hist(obs.windspeed, bins=np.arange(0, 20, 0.5), edgecolor='none',weights=np.ones(len(obs.windspeed))/len(obs.windspeed), density=False,alpha=0.5,label='CONUS')
plt.xlabel('Wind speed (m/s)')
plt.ylabel('Frequency')
plt.xticks([0,2,4,6,8,10,12,14,16,18,20])
plt.legend(loc='upper right',fontsize=10)
plt.savefig("/kfs2/projects/sfcwinds/obs windspeed CONUS.png")
plt.show()

plt.figure()
plt.hist(obs.winddirection, bins=np.arange(0, 360, 30), edgecolor='none',weights=np.ones(len(obs.winddirection))/len(obs.winddirection), density=False,alpha=0.5,label='CONUS')
plt.xlabel('Wind direction (deg)')
plt.ylabel('Frequency')
#plt.title('CA')
plt.xticks([0,90,180,270,360])
plt.legend(loc='upper right',fontsize=10)
plt.savefig("/kfs2/projects/sfcwinds/obs winddirection CONUS.png")
plt.show()

plt.figure()
plt.hist(obs.gust, bins=np.arange(0, 20, 0.5), edgecolor='none',weights=np.ones(len(obs.gust))/len(obs.gust), density=False,alpha=0.5,label='CONUS')
plt.xlabel('Gust wind speed (m/s)')
plt.ylabel('Frequency')
#plt.title('CA')
plt.xticks([0,2,4,6,8,10,12,14,16,18,20])
plt.legend(loc='upper right',fontsize=10)
plt.savefig("/kfs2/projects/sfcwinds/obs gust CONUS.png")
plt.show()

