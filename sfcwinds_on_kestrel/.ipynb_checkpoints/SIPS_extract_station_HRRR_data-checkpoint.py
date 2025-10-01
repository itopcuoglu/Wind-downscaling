import os
import pandas as pd
import xarray as xr
import glob
import matplotlib.pyplot as plt
import numpy as np
from Functions_sfcwinds import *

# Enable interactive backend when running in IPython, ignore in plain Python scripts (this is for data inspection with interactive plots)
try:
    if callable(globals().get("get_ipython", None)):
        get_ipython().run_line_magic("matplotlib", "widget")  # type: ignore[name-defined]
except Exception:
    pass


# Where to save the files
save_folder = "/kfs2/projects/sfcwinds/HRRR_station_data"
if not os.path.exists(save_folder):
    os.makedirs(save_folder)



# %% HRRR data
HRRR_folder = "/kfs2/projects/sfcwinds/HRRR"
area = "US_SW" # keep this (we only have HRRR data for the US southwest downloaded for the entire 2014-2024 period)
date_String = "*"   # keep * or select a date for testing

# Find all HRRR files for this date and area
file_pattern = os.path.join(HRRR_folder, f"hrrr_{area}_{date_String}*.nc")
file_list = sorted(glob.glob(file_pattern))

# One base HRRR file for getting lon and lat
if file_list:
    hrrr_base = xr.open_dataset(file_list[-1], chunks="auto")


#%% Filter stations we want so that we can loop over them

# Read metadata of observations (this includes the lon/lat of each station)
meta = pd.read_csv("/kfs2/projects/sfcwinds/observations/metadata_CONUS.csv")

# filter which data to read
base_dir = "/kfs2/projects/sfcwinds/observations/"
stations = ["NMC60"]   #  meta[meta.state == "NM"].station_id.values  # Test for one station, a state, or similar
                       #  eventually use all stations in the US southwest. Maybe filter by state: CA, AZ, NM, UT, NV, CO, TX, WY, OK, KS, NE, SD, OR, ID




# %% Loop over stations and extract closest HRRR location
for station in stations:

    row = meta[meta['station_id'] == station].iloc[0]
    lon = row['lon']
    lat = row['lat']
    if lon < 0:
        lon = lon + 360 # convert to 0-360 (HRRR convention)
    print(f"Station: {station}, Longitude: {lon}, Latitude: {lat}")

    # Find closest HRRR location for this station (maybe also the 4 sourrounding stations)
    closest, closest_lat, closest_lon, idx = find_closest_HRRR_loc(hrrr_base, [lat, lon]) 
    

    # get idx of closest (HRRR has x and y as dimensions, lon and lat are only variables)
    y_closest = idx[0]
    x_closest = idx[1]

    # loop over all HRRR files and extract data for this location
    # something like 
    hrrr = xr.open_mfdataset(file_list[:], concat_dim ="valid_time",combine='nested', chunks="auto", parallel = True)
    # but this is very unefficient (opening all HRRR files for each station)
    # maybe better open a HRRR file, extract data for all stations, append to each station file, then close the HRRR file and open the next one

    # use just the station locations, semthing like
    hrrr_loc = hrrr.sel(x = x_closest, y = y_closest) # include a check that HRRR lat and lon is not too far (3km?) away from the station
    
    # for testing: distance between station and closest HRRR grid point (should be below 1.5km at 3km grid spacing)
    # distance_m = haversine(hrrr_loc.latitude.values[-1], hrrr_loc.longitude.values[-1], lat, lon)


    # There will be data issues because the HRRR files changed slightly over time when donwloading from NOMADS
    # fix the data issues

    # correct the HRRR wind direction (model coordinate system into Noth-East)
    hrrr_loc["u10"], hrrr_loc["v10"] = rotate_to_true_north(hrrr_loc["u10"], hrrr_loc["v10"], hrrr_loc["longitude"])

    # calculate wind speed and direction
    hrrr_loc["wspd10"], hrrr_loc["wdir10"] = wspd_wdir_from_uv(hrrr_loc["u10"], hrrr_loc["v10"])

    # Make some plots comparing wind speed and direction for one / a few stations (compare HRRR to observations)
    # see below for code example

    # save HRRR data for this station
    file_path = os.path.join(save_folder, f"HRRR_{station}.nc")
    hrrr_loc.to_netcdf(file_path)  # there are some encoding options if file size or write speed is an issue, see the download scripts.



# # #Read observations according to station filter -= read the actual data

# print (f"Processing station {station}")
# all_dfs = []
# years    = ["2024"]  # for testing shorter time series
# for year in years:
#     pattern = os.path.join(base_dir, "*", station, f"{year}.parquet")
#     found = glob.glob(pattern)

#     for f in found:
#         try:
#             df = pd.read_parquet(f)
#             # Extract station_id from the path (e.g., .../CoAgMet/HOT01/2023.parquet)
#             station_id = os.path.basename(os.path.dirname(f))  # 'HOT01'
#             df["station_id"] = station_id
#             all_dfs.append(df)
#         except Exception as e:
#             print(f"Failed to read {f}: {e}")

# print(f"{len(all_dfs)} files found.")
# obs = pd.concat(all_dfs, ignore_index=True)
# obs["timestamp"] = obs["timestamp"].dt.tz_convert(None)

# # Plor obs vs. HRRR for this station
# fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

# # Wind speed
# axs[0].plot(obs.timestamp.values, obs.windspeed.values, ".", color="green", label="Obs "+station)
# axs[0].plot(hrrr_loc.valid_time.values, hrrr_loc.wspd10.values, ".", color="blue", label="HRRR")
# axs[0].set_ylabel("Wind Speed (m/s)")
# axs[0].legend()
# axs[0].grid(True)

# # Wind direction
# axs[1].plot(obs.timestamp.values, obs.winddirection.values, ".", color="green", label="Obs "+station)
# axs[1].plot(hrrr_loc.valid_time.values, hrrr_loc.wdir10.values, ".", color="blue", label="HRRR")
# axs[1].set_ylabel("Wind Direction (deg)")
# axs[1].set_xlabel("Time")
# axs[1].grid(True)

# plt.tight_layout()
# plt.show()






