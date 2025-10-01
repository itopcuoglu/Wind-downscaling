
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Enable interactive backend when running in IPython, ignore in plain Python scripts
try:
    if callable(globals().get("get_ipython", None)):
        get_ipython().run_line_magic("matplotlib", "widget")  # type: ignore[name-defined]
except Exception:
    pass

# Load the elevation dataset
ncfile = '/kfs2/projects/sfcwinds/environmental_data/CONUS_elevation_1km.nc'
ds = xr.open_dataset(ncfile)

# Extract variables
mean_elev = ds['elevation_mean']
lat = ds['lat']
lon = ds['lon']



# Define sites and their coordinates
sites = {
    'Tall Towers 42361': {'coords': (27.55, -92.49)},
    'Tall Towers 42362': {'coords': (27.8, -90.65)},
    'Tall Towers 42363': {'coords': (28.16, -89.22)},
    'Tall Towers 42364': {'coords': (29.06, -88.09)},
    'Tall Towers 42394': {'coords': (28.16, -89.24)},
    'Tall Towers Brookhaven': {'coords': (33.87, -84.34)},
    'NWTC M4': {'coords': (39.91, -105.23)},
    'NWTC M5': {'coords': (39.91, -105.23)},
    'Tall Towers Park Falls': {'coords': (45.95, -90.27)},
    'Tall Towers South Carolina': {'coords': (33.41, -81.83)},
    'Tall Towers upbc1': {'coords': (38.04, -122.12)},
    'Tall Towers Walnut Grove': {'coords': (38.26, -121.49)},
    'AWAKEN event log': {'coords': (36.362098, -97.405112)},
    'Hanford 400ft': {'coords': (46.56288408, -119.5993823)},
    'ETTI4': {'coords': (42.34575, -93.51945)},
    'Humboldt': {'coords': (40.9708, -124.5901)},
    'Morro': {'coords': (35.71074, -121.84606)},
    'ARM sgp': {'coords': (36.605295, -97.486581)},
    'ARM epc': {'coords': (32.866778, -117.25606)},
    'ARM guc': {'coords': (38.956101, -106.98783)},
    'ARM hou': {'coords': (29.67, -95.059)},
    'NYSERDA E05 Hudson North': {'coords': (39.97, -72.72)},
    'NYSERDA E05 Hudson Southwest': {'coords': (39.48, -73.59)},
    'NYSERDA E06 Hudson South': {'coords': (39.55, -73.43)},
    'WHOI': {'coords': (41.32704, -70.566678)},
}

# Plot mean elevation with Cartopy for US country and state outlines
fig = plt.figure(figsize=(10, 6))
ax = plt.axes(projection=ccrs.PlateCarree())
mesh = ax.pcolormesh(lon, lat, mean_elev, cmap='terrain', shading='auto', transform=ccrs.PlateCarree(),vmin = 0,vmax=4000)
plt.colorbar(mesh, ax=ax, label='Mean Elevation (m)')
ax.add_feature(cfeature.BORDERS, linewidth=1)
ax.add_feature(cfeature.STATES, linewidth=0.5)
ax.add_feature(cfeature.COASTLINE, linewidth=1)
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_title('Mean Elevation on 1km Grid')
ax.set_extent([lon.min(), lon.max(), lat.min(), lat.max()])

# Plot station locations and labels
for name, info in sites.items():
    lat_pt, lon_pt = info['coords']
    ax.scatter(lon_pt, lat_pt, color='red', s=40, marker='o', edgecolor='black', zorder=5, transform=ccrs.PlateCarree())
    ax.text(lon_pt, lat_pt, name, fontsize=7, ha='left', va='bottom', color='black', transform=ccrs.PlateCarree(), zorder=6)

plt.tight_layout()
plt.show()

