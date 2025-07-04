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
#states_SW = ["NM"]
states_SW = ["CA","TX","AZ","NM","CO","UT","NV","OK"]
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


plt.figure()
plt.hist(obs.height, bins=np.arange(0, 10, 0.1), edgecolor='none',weights=np.ones(len(obs.height))/len(obs.height), density=False,alpha=0.5,label='SW')
plt.xlabel('Height (m))')
plt.ylabel('Frequency')
#plt.title('CA')
plt.xticks([0,1,2,3,4,5,6,7,8,9,10])
plt.legend(loc='upper right',fontsize=10)
plt.savefig("obs height SW.png")
plt.show()

plt.figure()
plt.hist(obs.windspeed, bins=np.arange(0, 20, 0.1), edgecolor='none',weights=np.ones(len(obs.windspeed))/len(obs.windspeed), density=False,alpha=0.5,label='SW')
plt.xlabel('Wind speed (m/s))')
plt.ylabel('Frequency')
#plt.title('CA')
plt.xticks([0,2,4,6,8,10,12,14,16,18,20])
plt.legend(loc='upper right',fontsize=10)
plt.savefig("obs windspeed SW.png")
plt.show()

plt.figure()
plt.hist(obs.winddirection, bins=np.arange(0, 360, 0.1), edgecolor='none',weights=np.ones(len(obs.winddirection))/len(obs.winddirection), density=False,alpha=0.5,label='SW')
plt.xlabel('Wind direction (deg))')
plt.ylabel('Frequency')
#plt.title('CA')
plt.xticks([0,90,180,270,360])
plt.legend(loc='upper right',fontsize=10)
plt.savefig("obs winddirection SW.png")
plt.show()

plt.figure()
plt.hist(obs.gust, bins=np.arange(0, 20, 0.1), edgecolor='none',weights=np.ones(len(obs.gust))/len(obs.gust), density=False,alpha=0.5,label='SW')
plt.xlabel('Gust wind speed (m/s))')
plt.ylabel('Frequency')
#plt.title('CA')
plt.xticks([0,2,4,6,8,10,12,14,16,18,20])
plt.legend(loc='upper right',fontsize=10)
plt.savefig("obs gust SW.png")
plt.show()


