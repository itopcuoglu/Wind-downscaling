import pandas as pd
import xarray as xr
import glob
#matplotlib widget
import netCDF4
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow
from datetime import datetime, timedelta
import os
import sys
import warnings
warnings.filterwarnings("ignore", message="Will not remove GRIB file because it previously existed.")
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)

import dask
from dask.distributed import Client, LocalCluster
from dask import delayed, compute

def flush_print(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()
    sys.stderr.flush()

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

latitude_stations = obs.groupby('station_id')['lat'].apply(list)
longitude_stations = obs.groupby('station_id')['lon'].apply(list)
station_id = obs['station_id'].unique()   

# Read surfce roughness from an hourly file  - maybe later use the daily-current file, but surface toughness should not change that often.
#hrrr = xr.open_dataset("HRRR/hrrr_US_SW_2024-12-31.nc")   # can be any file
#hrrr = xr.open_dataset("C:/Users/memes/Documents/SIPS/HRRR US/hrrr_US_2024-12-30.nc")   # can be any file
#hrrr = xr.open_mfdataset('C:/Users/memes/Documents/SIPS/HRRR/*.nc', chunks={'time': 100})   # multiple files

HRRR_folder = "/kfs2/projects/sfcwinds/HRRR"
area = "US_SW"
year = "2024"
#date_string = "2024-12-30"


file_pattern = os.path.join(HRRR_folder, f"hrrr_{area}_{year}*.nc")
file_list = sorted(glob.glob(file_pattern))
if file_list:
    hrrr = xr.open_mfdataset(file_list[:], concat_dim ="valid_time",combine='nested', chunks="auto", parallel = True)
    
  
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

#hrrr.to_netcdf("/kfs2/projects/sfcwinds/scripts/hrrr.nc")


# Define compression settings
comp = {"zlib": True, "complevel": 4}  

# save file
encoding = {
    var: {**comp, "chunksizes": (12, 1, 200, 200)} # 
    if len(hrrr[var].dims) == 4 else {**comp, "chunksizes": (1, 200, 200)}
    for var in hrrr.data_vars
    }
hrrr.to_netcdf('/kfs2/projects/sfcwinds/scripts\\hrrr.nc', encoding=encoding)
