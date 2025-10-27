import os
import pandas as pd
import xarray as xr
import glob
import matplotlib.pyplot as plt
import numpy as np
from Functions_sfcwinds import *
#from nco import Nco
#nco=Nco()

# Enable interactive backend when running in IPython, ignore in plain Python scripts (this is for data inspection with interactive plots)
try:
    if callable(globals().get("get_ipython", None)):
        get_ipython().run_line_magic("matplotlib", "widget")  # type: ignore[name-defined]
except Exception:
    pass


# Where to save the files
#save_folder = "/kfs2/projects/sfcwinds/HRRR_station_data"
save_folder = "/scratch/itopcuog/sfcwinds/HRRR_station_data"
if not os.path.exists(save_folder):
    os.makedirs(save_folder)



# %% HRRR data
#HRRR_folder = "/kfs2/projects/sfcwinds/HRRR"
HRRR_folder = "/home/itopcuog/sfcwinds_dir/reduced_data/HRRR"
area = "US_SW" # keep this (we only have HRRR data for the US southwest downloaded for the entire 2014-2024 period)
date_String = "*"   # keep * or select a date for testing

# Find all HRRR files for this date and area
file_pattern = os.path.join(HRRR_folder, f"hrrr_{area}_{date_String}*.nc")
file_list = sorted(glob.glob(file_pattern))

# One base HRRR file for getting lon and lat
if file_list:
    hrrr_base = xr.open_dataset(file_list[-1], chunks="auto", decode_timedelta=True, engine='netcdf4')


#%% Filter stations we want so that we can loop over them

# Read metadata of observations (this includes the lon/lat of each station)
meta = pd.read_csv("/kfs2/projects/sfcwinds/observations/metadata_CONUS.csv")

# filter which data to read
base_dir = "/kfs2/projects/sfcwinds/observations/"
#stations = ["NMC60"]   #  meta[meta.state == "NM"].station_id.values  # Test for one station, a state, or similar
stations = ["NMC60", "NMC63"]   #  meta[meta.state == "NM"].station_id.values  # Test for one station, a state, or similar
                       #  eventually use all stations in the US southwest. Maybe filter by state: CA, AZ, NM, UT, NV, CO, TX, WY, OK, KS, NE, SD, OR, ID



# First record all the station locations
# The number of stations that are probed for
len_station=len(stations);
# Array to hold all the station locations
x_closest=[0] * len_station
y_closest=[0] * len_station


# Sweeping over all stations to determine the
# closest coordinates for each one
for station in stations:

    row = meta[meta['station_id'] == station].iloc[0]
    lon = row['lon']
    lat = row['lat']
    if lon < 0:
        lon = lon + 360 # convert to 0-360 (HRRR convention)
    print(f"Station: {station}, Longitude: {lon}, Latitude: {lat}")

    # Find closest HRRR location for this station (maybe also the 4 sourrounding stations)
    closest, closest_lat, closest_lon, idx = find_closest_HRRR_loc(hrrr_base, [lat, lon]) 
    print(f"Coordinates are: {closest_lat}, {closest_lon}")


    # get idx of the closest measurement point for each station(HRRR has x and y as dimensions, lon and lat are only variables)
    st_ind=stations.index(station);
    print(f"Station index for {station} is {st_ind}")
    y_closest[st_ind] = idx[0]
    x_closest[st_ind] = idx[1]
    print(f"station {station}, xc {x_closest[st_ind]}, yc {y_closest[st_ind]}")

    # loop over all HRRR files and extract data for this location
    # something like 

print(f"File list: {file_list[:]}")

# Sweep over all files, one file at a time
for ifile in file_list:
    print(f"filenameis {ifile}")
    # Opening ifile, which is only one file
    hrrr = xr.open_dataset(ifile, chunks="auto", engine='netcdf4', decode_timedelta= True)

# Sweep over all stations within each file loop
# This way, only one file is open whiledata for several
# stations are being extracted
    for station in stations:
        print(f"Station name is {station}")
        st_ind=stations.index(station)
        print(f"Station index is {st_ind}")
        hrrr_loc = hrrr.isel(x = x_closest[st_ind], y = y_closest[st_ind]) # include a check that HRRR lat and lon is not too far (3km?) away from the station
        print(f"Location for station {station} is {x_closest[st_ind]} {y_closest[st_ind]}") 
    # for testing: distance between station and closest HRRR grid point (should be below 1.5km at 3km grid spacing)
        #distance_m = haversine(hrrr_loc.latitude.values[-1], hrrr_loc.longitude.values[-1], lat, lon)
        #print(f"Distance is {distance_m}")

    # correct the HRRR wind direction (model coordinate system into Noth-East)
        hrrr_loc["u10"], hrrr_loc["v10"] = rotate_to_true_north(hrrr_loc["u10"], hrrr_loc["v10"], hrrr_loc["longitude"])

    # calculate wind speed and direction
        hrrr_loc["wspd10"], hrrr_loc["wdir10"] = wspd_wdir_from_uv(hrrr_loc["u10"], hrrr_loc["v10"])

    # Make some plots comparing wind speed and direction for one / a few stations (compare HRRR to observations)
    # see below for code example
    # This is yet to be done properly

    # save HRRR data for the current station
        # Extracting the date suffix from each hrrr file name
        sub_str=ifile[-13:-3]
        print(f"Subtracted string {sub_str}")
        # Appending the extracted string to the station name
        # so that each station can have individual files for each date
        # that are to be merged later
        file_path = os.path.join(save_folder, f"HRRR_{station}_{sub_str}.nc")
        hrrr_loc.to_netcdf(file_path,mode='a')  # there are some encoding options if file size or write speed is an issue, see the download scripts.

    hrrr.close()


# Merge the individual files for each station
for station in stations:
    date_String = "*"   # wildcard for dates
    file_part_pattern = os.path.join(save_folder, f"HRRR_{station}_{date_String}*.nc")
    file_part_list = sorted(glob.glob(file_part_pattern))
    # Load the datewise separated files for one station only
    # using open_mfdataset
    # This allows for concatenating over valid_time
    concat_data = xr.open_mfdataset(file_part_list[:], concat_dim ="valid_time",combine='nested', chunks="auto", parallel = True, engine='netcdf4', decode_timedelta= True)
    total_output_path = os.path.join(save_folder, f"HRRR_{station}.nc")
    concat_data.to_netcdf(total_output_path,mode='w')  
    

# # #Read observations according to station filter -= read the actual data

# print (f"Processing station {station}")
all_dfs = []
 #years    = ["2024"]  # for testing shorter time series
years    = ["2025"]  # for testing shorter time series
for year in years:
    pattern = os.path.join(base_dir, "*", station, f"{year}.parquet")
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

    print(f"{len(all_dfs)} files found.")
    obs = pd.concat(all_dfs, ignore_index=True)
    obs["timestamp"] = obs["timestamp"].dt.tz_convert(None)

# # Plor obs vs. HRRR for this station
    fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

# # Wind speed
    axs[0].plot(obs.timestamp.values, obs.windspeed.values, ".", color="green", label="Obs "+station)
    axs[0].plot(hrrr_loc.valid_time.values, hrrr_loc.wspd10.values, ".", color="blue", label="HRRR")
    axs[0].set_ylabel("Wind Speed (m/s)")
    axs[0].legend()
    axs[0].grid(True)

# # Wind direction
    axs[1].plot(obs.timestamp.values, obs.winddirection.values, ".", color="green", label="Obs "+station)
    axs[1].plot(hrrr_loc.valid_time.values, hrrr_loc.wdir10.values, ".", color="blue", label="HRRR")
    axs[1].set_ylabel("Wind Direction (deg)")
    axs[1].set_xlabel("Time")
    axs[1].grid(True)

    plt.tight_layout()
    plt.savefig(f"{year}.png")
    #plt.show()






