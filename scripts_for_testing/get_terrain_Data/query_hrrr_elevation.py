import xarray as xr
import numpy as np

def find_closest_grid_point(lon, lat, lon_grid, lat_grid):
    # Compute distance to all grid points
    dist = np.sqrt((lon_grid - lon)**2 + (lat_grid - lat)**2)
    idx = np.unravel_index(np.argmin(dist), dist.shape)
    return idx

def get_vicinity_mean_std(std_array, idx, radius=3):
    y, x = idx
    y1 = max(0, y - radius)
    y2 = min(std_array.shape[0], y + radius + 1)
    x1 = max(0, x - radius)
    x2 = min(std_array.shape[1], x + radius + 1)
    vicinity = std_array[y1:y2, x1:x2]
    return np.nanmean(vicinity)

# User input
query_lon = -112.0  # example longitude
query_lat = 26.5    # example latitude

# Load data
ncfile = 'CONUS_elevation_1km.nc'
ds = xr.open_dataset(ncfile)
mean_elev = ds['elevation_mean'].values
std_elev = ds['elevation_std'].values
lat_grid = ds['lat'].values
lon_grid = ds['lon'].values

# Find closest grid point
idx = find_closest_grid_point(query_lon, query_lat, lon_grid, lat_grid)
closest_elev = mean_elev[idx]
mean_std_vicinity = get_vicinity_mean_std(std_elev, idx, radius=3)

print(f"Closest grid point index: {idx}")
print(f"Elevation at closest grid point: {closest_elev:.2f} m")
print(f"Mean of std in 9km vicinity: {mean_std_vicinity:.2f} m")
