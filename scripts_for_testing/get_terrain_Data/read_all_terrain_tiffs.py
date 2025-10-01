
import os
import glob
import numpy as np
import rasterio
# import matplotlib
# matplotlib.use('QtAgg') # Or 'TkAgg', 'GTK3Agg', etc.
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import concurrent.futures
import time



# Enable interactive backend when running in IPython, ignore in plain Python scripts
try:
    if callable(globals().get("get_ipython", None)):
        get_ipython().run_line_magic("matplotlib", "widget")  # type: ignore[name-defined]
except Exception:
    pass


# Directory containing the .tif files
TIF_DIR = os.path.join(os.path.dirname(__file__), 'CONUS_terrain_files')

def process_tif(tif_file):
    with rasterio.open(tif_file) as src:
        arr = src.read(1)
        arr = np.where(arr == -9.9999900e+05, np.nan, arr)
        rows, cols = arr.shape
        row_indices = np.arange(rows)
        col_indices = np.arange(cols)
        lon_1d, _ = rasterio.transform.xy(src.transform, 0, col_indices, offset='center')
        _, lat_1d = rasterio.transform.xy(src.transform, row_indices, 0, offset='center')
        arr_min = np.nanmin(arr)
        arr_max = np.nanmax(arr)
        lon_min = np.nanmin(lon_1d)
        lon_max = np.nanmax(lon_1d)
        lat_min = np.nanmin(lat_1d)
        lat_max = np.nanmax(lat_1d)
        return {
            'arr': arr,
            'lon_1d': np.array(lon_1d),
            'lat_1d': np.array(lat_1d),
            'arr_min': arr_min,
            'arr_max': arr_max,
            'lon_min': lon_min,
            'lon_max': lon_max,
            'lat_min': lat_min,
            'lat_max': lat_max
        }

def plot_all_tiffs(tif_dir, max_files=5):
    start_time = time.time()
    tif_files = glob.glob(os.path.join(tif_dir, '*.tif'))[:max_files]
    fig = plt.figure(figsize=(10, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    # Use multiprocessing to process files in parallel
    results = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(process_tif, tif_files))
    # Aggregate results
    vmin = min(r['arr_min'] for r in results)
    vmax = max(r['arr_max'] for r in results)
    lon_min = min(r['lon_min'] for r in results)
    lon_max = max(r['lon_max'] for r in results)
    lat_min = min(r['lat_min'] for r in results)
    lat_max = max(r['lat_max'] for r in results)
    # Plot all arrays
    mesh = None
    for r in results:
        mesh = ax.pcolormesh(r['lon_1d'], r['lat_1d'], r['arr'], cmap='terrain', shading='auto', vmin=vmin, vmax=vmax)
    plt.colorbar(mesh, ax=ax, label='Elevation (m)')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Combined Elevation Map of All Tiles')
    ax.add_feature(cfeature.BORDERS, linewidth=1)
    ax.add_feature(cfeature.STATES, linewidth=0.5)
    ax.add_feature(cfeature.COASTLINE, linewidth=1)
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    plt.tight_layout()
    plt.show()
    elapsed = time.time() - start_time
    print(f"Elapsed time: {elapsed:.2f} seconds")

if __name__ == '__main__':
    plot_all_tiffs(TIF_DIR, max_files=5)
