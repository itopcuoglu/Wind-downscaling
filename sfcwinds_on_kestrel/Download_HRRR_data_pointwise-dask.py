from herbie import Herbie, FastHerbie
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import shapely.geometry as sgeom
import cartopy.feature
import cartopy.feature as cfeature
from cartopy.feature import ShapelyFeature
from cartopy.io.shapereader import Reader
import numpy as np
import pandas as pd
import glob
import matplotlib.cm as cm
import xarray as xr
from datetime import datetime, timedelta
import time
import os
import warnings
from dask import delayed, compute
from dask.distributed import Client, LocalCluster

warnings.filterwarnings("ignore", message="Will not remove GRIB file because it previously existed.")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*decode_timedelta.*")

"""
HRRR model documentation:
https://rapidrefresh.noaa.gov/hrrr/
Documentation for the HRRR data download wrapper:
https://herbie.readthedocs.io/en/stable/
Variables:
https://www.nco.ncep.noaa.gov/pmb/products/hrrr/hrrr.t00z.wrfnatf00.grib2.shtml
"""

#%% Define

# Folder to save datasets
save_folder = "/kfs2/projects/sfcwinds/HRRR_pointwise_new"

G3P3 = (-106.510100,  34.962400, "G3P3")

# date range with one-hour frequency, reversed (starts from end)
date_range = pd.date_range(datetime(2025, 3, 21), 
                           datetime(2025, 6, 3), 
                           freq="h").tolist()[::-1]

# Forecast hour(s)
fxx = list(range(0,18))

# Define synoptic hours (48h forecast range for hourly files at these times)
synoptic_hours = [0, 6, 12, 18]

# Define compression settings
comp = {"zlib": True, "complevel": 4}  

#%% Functions

def safe_xarray(herbie_obj, search_str):
    """Accessing the Herbie object sometimes gives a permission error, adding a wait time helps."""
    for _ in range(5):  # Retry up to 5 times
        try:
            return herbie_obj.xarray(search_str, remove_grib=True)
        except PermissionError:
            time.sleep(1)  # Wait and retry
    raise Exception(f"Failed to load {search_str} after multiple attempts.")

def process_hourly_data(timestamp, loc, fxx):
    """Retrieve, process, and return hourly HRRR dataset for a single timestamp and location."""
    max_retries = 10  # Set a limit to avoid infinite loops
    attempt = 0
    while attempt < max_retries:
        try:
            # Forecast range 18h or 48h depending on init time   (just for hourly, subhourly is always just 18h)
            # forecast_range = 49 if timestamp.hour in synoptic_hours else 19
            # fxx = list(range(forecast_range))
            
            H_subh = FastHerbie([timestamp], model='hrrr', product='subh', verbose=False, fxx = fxx)
            df = H_subh.inventory()[["variable", "search_this"]]
            filtered_df_10 = df[df["search_this"].str.contains(r"\b(?:ugrd|vgrd):10\b", case=False, na=False)
                                & ~df["search_this"].str.contains("ave", case=False, na=False)]
            wind_search_string_10 = "|".join(filtered_df_10["search_this"].drop_duplicates().values)
            
            filtered_df_80 = df[df["search_this"].str.contains(r"\b(?:ugrd|vgrd):80\b", case=False, na=False)
                                & ~df["search_this"].str.contains("ave", case=False, na=False)]
            wind_search_string_80 = "|".join(filtered_df_80["search_this"].drop_duplicates().values)

            # Subhourly data
            ds_subh = xr.merge([
                safe_xarray(H_subh, wind_search_string_10),
                safe_xarray(H_subh, wind_search_string_80).rename({"u": "u80", "v": "v80"}),
                safe_xarray(H_subh, ':TMP:2 m|DPT'),
                safe_xarray(H_subh, 'GUST')
            ], compat='minimal')
            
            # Select location
            points = pd.DataFrame({
                "latitude": [loc[1]],
                "longitude": [loc[0]],
                "stid": [loc[2]],
            })
            hrrr = ds_subh.herbie.pick_points(points).isel(point=0)
            
            # Cleanup variables and attributes
            hrrr = hrrr.drop_vars(
                ["boundaryLayerCloudLayer", "heightAboveGroundLayer", "metpy_crs", "surface", 
                 "gribfile_projection", "point", "point_latitude", "point_longitude", "point_stid", 
                 "point_grid_distance"], errors="ignore"
            )
            
            # Add the valid_time coordinate
            hrrr = hrrr.assign_coords(valid_time=hrrr.time + hrrr.step)
            
            # Add wind speed and direction
            hrrr['wspd10'] = (hrrr.u10**2 + hrrr.v10**2)**0.5
            hrrr['wdir10'] = np.degrees(np.arctan2(hrrr.u10, hrrr.v10))
            hrrr['wspd80'] = (hrrr.u80**2 + hrrr.v80**2)**0.5
            hrrr['wdir80'] = np.degrees(np.arctan2(hrrr.u80, hrrr.v80))
            return hrrr
        except Exception as e:
            print(f"Error: {e}")
            attempt += 1
            print(f"Attempt {attempt} failed. Retrying...")
            time.sleep(1)
    return None

@delayed
def process_and_save(timestamp, loc, fxx, save_folder, comp):
    hourly_file_path = os.path.join(save_folder, f"hrrr_{loc[2]}_{timestamp.date()}_{timestamp.hour}h.nc")
    if os.path.exists(hourly_file_path):
        print(f"File already exists: {hourly_file_path}.")
        return "Exists"
    hrrr = process_hourly_data(timestamp, loc=loc, fxx=fxx)
    if hrrr is not None:
        encoding = {var: {**comp, "chunksizes": (1,)} for var in hrrr.data_vars}
        hrrr.to_netcdf(hourly_file_path)
        print(f"Saved hourly file: {hourly_file_path}")
        return "Saved"
    else:
        print(f"Failed to process: {hourly_file_path}")
        return "Failed"

def process_in_batches(dates, batch_size=100):
    """Process delayed tasks in batches to avoid overwhelming the Dask scheduler."""
    all_results = []
    for i in range(0, len(dates), batch_size):
        batch = dates[i:i+batch_size]
        delayed_tasks = [process_and_save(date, G3P3, fxx, save_folder, comp) for date in batch]
        results = compute(*delayed_tasks)
        all_results.extend(results)
        print(f"Processed batch {i//batch_size + 1} ({len(batch)} dates)")
    return all_results

if __name__ == "__main__":

    # Process in batches for large date ranges
    results = process_in_batches(date_range, batch_size=100)
    print("All results:", results)
