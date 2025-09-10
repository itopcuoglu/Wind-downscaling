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
from datetime import datetime

client = cdsapi.Client()  # kept for backward compatibility; parallel path builds clients per task

def flush_print(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()
    sys.stderr.flush()


### Definitions

# Download data at surface level of pressure levels
data_type = "land"  # select "surface" or "pressure_level" or "land"

# Time frame to download data

start = "2000-01-01"
end = "2023-12-31"

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

# Concurrency controls
DEFAULT_WORKERS = int(os.environ.get("ERA5_DASK_WORKERS", "4"))
MAX_RETRIES = int(os.environ.get("ERA5_DOWNLOAD_RETRIES", "5"))
INITIAL_BACKOFF = float(os.environ.get("ERA5_BACKOFF_SECS", "10"))


def build_request_for_date(data_type: str, date: pd.Timestamp):
    """Build CDS dataset, request dict, and target file for a given date and data_type."""
    year = date.strftime('%Y')
    month = date.strftime('%m')
    day = date.strftime('%d')

    if data_type == "surface":
        dataset = "reanalysis-era5-single-levels"
        daily_file_path = os.path.join(save_path, f'ERA5_surface_{year}_{month}_{day}.nc')
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
                "standard_deviation_of_orography",
            ],
            'year': [year],
            'month': [month],
            'day': [day],
            'time': times,
            'area': download_area,
            "data_format": "netcdf",
        }
    elif data_type == "pressure_level":
        dataset = 'reanalysis-era5-pressure-levels'
        daily_file_path = os.path.join(save_path, f'ERA5_pressure_{year}_{month}_{day}.nc')
        request = {
            'product_type': 'reanalysis',
            "variable": [
                "geopotential",
                "relative_humidity",
                "temperature",
                "u_component_of_wind",
                "v_component_of_wind",
            ],
            "pressure_level": [
                "300", "400", "500",
                "600", "700", "800",
                "900", "925", "950",
                "975", "1000",
            ],
            'year': [year],
            'month': [month],
            'day': [day],
            'time': times,
            'area': download_area,
            "data_format": "netcdf",
        }
    elif data_type == "land":
        dataset = "reanalysis-era5-land"
        daily_file_path = os.path.join(save_path, f'ERA5_land_{year}_{month}_{day}.nc')
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
                # "snow_cover",
                # "forecast_albedo",
                "10m_u_component_of_wind",   # seems like 100m winds are not available
                "10m_v_component_of_wind",
                "surface_pressure",
                # "total_precipitation",
                # "high_vegetation_cover",
                # "low_vegetation_cover",
                # "geopotential",
                # # "land_sea_mask",
                # "soil_type",
                # "type_of_high_vegetation",
                # "type_of_low_vegetation",
            ],
        }
    else:
        raise ValueError(f"Unsupported data_type: {data_type}")

    return dataset, request, daily_file_path


def download_one(date: pd.Timestamp, max_retries: int = MAX_RETRIES) -> str:
    """Download a single day's file if missing. Returns status string."""
    dataset, request, target = build_request_for_date(data_type, date)

    # Skip if exists
    if os.path.exists(target):
        flush_print(f"Exists, skipping: {os.path.basename(target)}")
        return "exists"

    os.makedirs(os.path.dirname(target), exist_ok=True)
    attempt = 0
    backoff = INITIAL_BACKOFF
    start_time = time.perf_counter()
    while attempt < max_retries:
        try:
            # A fresh client per task is safer for concurrency
            cds = cdsapi.Client()  # rely on ~/.cdsapirc; avoid global client in threads/processes
            cds.retrieve(dataset, request, target)
            elapsed = time.perf_counter() - start_time
            flush_print(f"OK {os.path.basename(target)} in {elapsed:.1f}s")
            return "ok"
        except Exception as e:
            attempt += 1
            if attempt >= max_retries:
                flush_print(f"FAIL {os.path.basename(target)} after {attempt} attempts: {e}")
                return "fail"
            flush_print(f"Retry {attempt}/{max_retries} for {os.path.basename(target)} in {backoff:.0f}s due to: {e}")
            try:
                time.sleep(backoff)
            except KeyboardInterrupt:
                raise
            backoff = min(backoff * 1.8, 600)  # cap backoff at 10 minutes


def main():
    flush_print(f"Starting ERA5 downloads with data_type={data_type}, range {start}..{end}")

    # Prefilter dates to download to reduce scheduler load
    pending_dates = []
    for d in dates:
        _, _, target = build_request_for_date(data_type, d)
        if not os.path.exists(target):
            pending_dates.append(d)

    total = len(pending_dates)
    if total == 0:
        flush_print("All requested files already present. Nothing to do.")
        return

    flush_print(f"Pending days: {total}. Using up to {DEFAULT_WORKERS} workers.")

    # Thread pool parallel downloads
    try:
        from concurrent.futures import ThreadPoolExecutor, as_completed as futures_as_completed
        with ThreadPoolExecutor(max_workers=DEFAULT_WORKERS) as ex:
            futs = [ex.submit(download_one, d) for d in pending_dates]
            done = ok = fail = exist = 0
            for f in futures_as_completed(futs):
                res = f.result()
                done += 1
                if res == "ok":
                    ok += 1
                elif res == "exists":
                    exist += 1
                else:
                    fail += 1
                if done % 10 == 0 or done == total:
                    flush_print(f"Progress: {done}/{total} (ok={ok}, fail={fail})")
    except Exception as e:
        flush_print(f"Parallel fallback failed ({e}); running serially.")
        for d in pending_dates:
            download_one(d)


if __name__ == "__main__":
    main()


### Download data now handled via Dask-enabled main()