import cdsapi
import zipfile
import xarray as xr
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
import sys
import time

client = cdsapi.Client() # timeout=6000,debug=False)

def flush_print(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()
    sys.stderr.flush()


### Definitions

# Download data at surface level of pressure levels
data_type = "surface"  # select "surface" or "pressure_level" or "land"

# Time frame to download data

start = "2023-01-01"
end = "2024-12-31"

dates = pd.date_range(start=start, end=end, freq="D")[::-1]  # Reverse order

times = [
        "00:00", "01:00", "02:00",
        "03:00", "04:00", "05:00",
        "06:00", "07:00", "08:00",
        "09:00", "10:00", "11:00",
        "12:00", "13:00", "14:00",
        "15:00", "16:00", "17:00",
        "18:00", "19:00", "20:00",
        "21:00", "22:00", "23:00"
    ]

# Spatial range to download data
download_area = [53, -132, 20, -60]   # this is CONUS


# Path for saving data
save_path = '/kfs2/projects/sfcwinds/ERA5/'


### Download data
if data_type == "surface":

    dataset = "reanalysis-era5-single-levels"
    
    for date in dates:   # downloading files in a loop, because otherwise all data would be stored in a single file.

        start_time = time.perf_counter()
        
        year = date.strftime('%Y')
        month = date.strftime('%m')
        day = date.strftime('%d')
            
        flush_print(year + "-" + month + "-" + day)

        daily_file_path = save_path + f'ERA5_surface_{year}_{month}_{day}.nc'

        # Check if file already exists, exit loop if so
        if os.path.exists(daily_file_path):
            flush_print(f"File already exists: {daily_file_path}.")
            continue

        request = {
                'product_type': 'reanalysis',
                "variable": [
                    "10m_u_component_of_wind",
                    "10m_v_component_of_wind",
                    "2m_dewpoint_temperature",
                    "2m_temperature",
                    "mean_sea_level_pressure",
                    "sea_surface_temperature",
                    "surface_pressure",
                    "100m_u_component_of_wind",
                    "100m_v_component_of_wind",
                    "10m_u_component_of_neutral_wind",
                    "10m_v_component_of_neutral_wind",
                    # "10m_wind_gust_since_previous_post_processing",  # this parameter somehow messes up the nc file to be unreadable!
                    "instantaneous_10m_wind_gust",
                    # "surface_latent_heat_flux",  # this parameter somehow messes up the nc file to be unreadable!
                    "high_vegetation_cover",
                    "leaf_area_index_high_vegetation",
                    "leaf_area_index_low_vegetation",
                    "low_vegetation_cover",
                    "type_of_high_vegetation",
                    "type_of_low_vegetation",
                    "angle_of_sub_gridscale_orography",
                    "anisotropy_of_sub_gridscale_orography",
                    "convective_available_potential_energy",
                    "forecast_surface_roughness",
                    "friction_velocity",
                    "geopotential",
                    "instantaneous_moisture_flux",
                    "k_index",
                    "land_sea_mask",
                    "standard_deviation_of_orography"
                ],
                'year': [year],
                'month': [month],
                'day': [day],
                'time': times,
                'area': download_area,
                "data_format": "netcdf",
            }

        target = daily_file_path
    
        client.retrieve(dataset, request, target)


        # Timing
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time:.4f} seconds")



elif data_type == "pressure_level":

    dataset = 'reanalysis-era5-pressure-levels'
    
    for date in dates:   # downloading files in a loop, because otherwise all data would be stored in a single file.
        year = date.strftime('%Y')
        month = date.strftime('%m')
        day = date.strftime('%d')
            
        flush_print(year + "-" + month + "-" + day)

        daily_file_path = save_path + f'ERA5_pressure_{year}_{month}_{day}.nc'

        # Check if file already exists, exit loop if so
        if os.path.exists(daily_file_path):
            flush_print(f"File already exists: {daily_file_path}.")
            continue

        request = {
                'product_type': 'reanalysis',
                "variable": [
                    "geopotential",
                    "relative_humidity",
                    "temperature",
                    "u_component_of_wind",
                    "v_component_of_wind"
                ],
                "pressure_level": [
                    "300", "400", "500",
                    "600", "700", "800",
                    "900", "925", "950",
                    "975", "1000"
                ],
                'year': [year],
                'month': [month],
                'day': [day],
                'time': times,
                'area': download_area,
                "data_format": "netcdf",
            }

        target = daily_file_path
    
        client.retrieve(dataset, request, target)



elif data_type == "land":   # ERA5-Land with 9km resolution 1950-present

    dataset = "reanalysis-era5-land"
    
    for date in dates:   # downloading files in a loop, because otherwise all data would be stored in a single file.
        year = date.strftime('%Y')
        month = date.strftime('%m')
        day = date.strftime('%d')
            
        flush_print(year + "-" + month + "-" + day)

        daily_file_path = save_path + f'ERA5_land_{year}_{month}_{day}.nc'

        # Check if file already exists, exit loop if so
        if os.path.exists(daily_file_path):
            flush_print(f"File already exists: {daily_file_path}.")
            continue

        request = {
                "year": [year],
                "month": [month],
                "day": [day],
                "time": times,
                "data_format": "netcdf",
                "download_format": "unarchived",
                "area": download_area,
                "variable": [
                    "2m_dewpoint_temperature",
                    "2m_temperature",
                    #"snow_cover",
                   # "forecast_albedo",
                    "10m_u_component_of_wind",   # seems like 100m winds are not available
                    "10m_v_component_of_wind",
                     "surface_pressure",
                    # "total_precipitation",   # Some of the commented parameters mess up the ncfile, but they are not really needed anyways.
                    # "high_vegetation_cover",
                    # "low_vegetation_cover",
                    # "geopotential",
                    # #"land_sea_mask",
                    # "soil_type",
                    # "type_of_high_vegetation",
                    # "type_of_low_vegetation"
                ]
            }

        target = daily_file_path
    
        client.retrieve(dataset, request, target)