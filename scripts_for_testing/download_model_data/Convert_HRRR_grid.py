from pyresample import geometry, kd_tree




import metpy
from metpy.interpolate import interpolate_to_grid
from metpy.cbook import get_test_data
from metpy.units import units

# Parse coordinates
ds = ds.metpy.parse_cf()

# Get lat/lon values
lats = ds.latitude.values
lons = ds.longitude.values


# Define new lat/lon grid (e.g., 0.05° resolution)
lat_new = np.arange(lats.min().round(0), lats.max().round(0), 0.05)
lon_new = np.arange(lons.min().round(0), lons.max().round(0), 0.05)
lon_grid, lat_grid = np.meshgrid(lon_new, lat_new)

var = ds['u10'].isel(step=1).values  # Adjust variable name as needed




# Define the HRRR grid
hrrr_grid = geometry.SwathDefinition(lons=lons, lats=lats)

# Define the new lat/lon grid
target_grid = geometry.GridDefinition(lons=lon_grid, lats=lat_grid)

# Resample using nearest-neighbor
var_interp = kd_tree.resample_nearest(hrrr_grid, var, target_grid, radius_of_influence=50000)




# Create projection




import cartopy.crs as ccrs

projection = ccrs.LambertConformal(central_longitude=262.5, 
                                   central_latitude=38.5, 
                                   standard_parallels=(38.5, 38.5),
                                    globe=ccrs.Globe(semimajor_axis=6371229,
                                                     semiminor_axis=6371229))







def check_boundaries(data):
    
    lat_top = 39
    lat_bottom = 34
    lon_top = -107
    lon_bottom = -110 # Four Corners region
    return ((lat_bottom < data.latitude) & (data.latitude < lat_top) & (
        lon_bottom < data.longitude) & (data.longitude < lon_top)).compute()

area = ds.where(check_boundaries, drop=True)


