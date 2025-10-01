import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Functions_sfcwinds import *



# Read surfce roughness from an hourly file  - maybe later use the daily-current file, but surface toughness should not change that often.
# hrrr = xr.open_dataset("/kfs2/projects/sfcwinds/test_case/HRRR_NM_2024-05-01_2024-05-31.nc")   # can be any file
hrrr = xr.open_dataset("/kfs2/projects/sfcwinds/HRRR/hrrr_US_SW_2024-01-01.nc")   # can be any file, but daily file is smaller than NM test case

hrrr['longitude'] = xr.where(hrrr['longitude'] > 180, hrrr['longitude'] - 360, hrrr['longitude'])

G3P3 = (-106.510100,  34.962400, "G3P3")

hrrr_loc, closest_lat, closest_lon, idx = find_closest_HRRR_loc(hrrr=hrrr, 
                                                                   hrrr_coords_to_use=[G3P3[1], G3P3[0]])


from pyproj import Geod

# G3P3 coordinates (lon, lat)
g3p3_lon, g3p3_lat = G3P3[0], G3P3[1]

# Closest HRRR grid coordinates
# grid_lon, grid_lat = closest_lon, closest_lat
grid_lon, grid_lat = hrrr_loc.longitude.values.mean(), hrrr_loc.latitude.values.mean()

# Calculate geodesic distance in meters
geod = Geod(ellps="WGS84")
_, _, distance_m = geod.inv(g3p3_lon, g3p3_lat, grid_lon, grid_lat)

print(f"Distance between G3P3 and HRRR grid point: {distance_m:.2f} meters")