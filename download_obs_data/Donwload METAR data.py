import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import shapely.geometry as sgeom
import cartopy.feature
import cartopy.feature as cfeature
from cartopy.feature import ShapelyFeature
from cartopy.io.shapereader import Reader
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
#from toolbox.cartopy_tools import common_features, pc
import pandas as pd
import s3fs
import numpy as np
import pyarrow
import glob
from windrose import WindroseAxes
import matplotlib.cm as cm
import xarray as xr
from datetime import datetime, timedelta


"""




"""


# Read Metar data from different station
def read_metar(network, start_year, start_month, start_day, end_year, end_month, end_day):
    
    """
    use: 
        
    met = read_metar(station='TPH', 
                       start_year=year, start_month=month, start_day=day, 
                       end_year=day_after.year, end_month=day_after.month, end_day=day_after.day)
    
    year, month, date can be either strings or integers
    """
        
    import requests
    from io import StringIO
    
    # API endpoint (https://mesonet.agron.iastate.edu/request/download.phtml?network=NV_ASOS)
    url = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
    params = {
        #'data': 'tmpf,dwpf,relh,drct,sknt,p01i,alti,mslp,vsby,gust,peak_wind_gust,peak_wind_drct,peak_wind_time',
        'data': 'drct,sknt,gust',
        #'station': station,
        'network': network,
        'latlon': "yes",
        'elev': "yes",
        'tz': 'UTC',
        'year1': start_year,
        'month1': start_month,
        'day1': start_day,
        'year2': end_year,
        'month2': end_month,
        'day2': end_day
    }
    
    # Send a GET request to the API
    response = requests.get(url, params=params)
    
    # Check if the request was successful
    if response.status_code == 200:
        # Load the response content into a DataFrame
        csv_data = StringIO(response.text)
        metar = pd.read_csv(csv_data, delimiter=',', index_col=1, parse_dates=True, na_values="M")
    else:
        metar = pd.DataFrame()
        print(f"Failed to retrieve Metar data: {response.status_code}")
        
    return metar




for network in ['NM_ASOS']:

    metar = read_metar(network=network, 
                       start_year=2021, start_month=11, start_day=1, 
                       end_year=2021, end_month=11, end_day=28)
    
    
    # Extract station locations and elevations
    stations = metar[['station', 'lon', 'lat', 'elevation']].drop_duplicates()
    
    # Ensure valid is a datetime index
    metar = metar.drop_duplicates().reset_index().set_index(["valid", "lon", "lat"])
    
    # Average any duplicates (you can also choose other methods like .first(), .last(), etc.)
    metar = metar.groupby(['valid', "lon", "lat"]).first()
    
    metar["wspd"] =metar.sknt/1.94384
    
    # Convert to xarray
    metar_xr = metar.to_xarray()
    
    # Rename dimensions for clarity
    metar_xr = metar_xr.rename({"valid": "time"})
    
    metar_xr.to_netcdf(f"../data/Metar/Metar_{network}_{metar_xr.time[0].values.astype('datetime64[D]')}_{metar_xr.time[-1].values.astype('datetime64[D]')}.nc")
    
    
    
    
    




# Create a figure with a map projection
fig, ax = plt.subplots(figsize=(10, 6), subplot_kw={'projection': ccrs.PlateCarree()})


# Add map features
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.add_feature(cfeature.STATES, linestyle='-', linewidth=0.5, edgecolor="gray") 
ax.add_feature(cfeature.LAND, edgecolor='black', alpha=0.3)
ax.add_feature(cfeature.LAKES, edgecolor='black', alpha=0.3)
ax.add_feature(cfeature.RIVERS, edgecolor='blue', alpha=0.3)
terrain = cfeature.NaturalEarthFeature(
    category='physical',
    name='shaded_relief',
    scale='110m'  # High resolution
)
ax.add_feature(terrain)  

# Scatter plot of stations with elevation color-coded
# sc = ax.scatter(
#     stations['lon'], stations['lat'], 
#     c=stations['elevation'], cmap='terrain', edgecolors='black', 
#     s=50, transform=ccrs.PlateCarree()
# )



# Flatten the data and remove NaN values
lon, lat = np.meshgrid(metar_xr['lon'], metar_xr['lat'])
lon_flat = lon.flatten()
lat_flat = lat.flatten()
elev_flat = metar_xr.mean(dim="time")['sknt'].T.values.flatten()

# Mask out NaN values
mask = ~np.isnan(elev_flat)
lon_clean = lon_flat[mask]
lat_clean = lat_flat[mask]
elev_clean = elev_flat[mask]

sc = plt.scatter(lon_clean, lat_clean, c=elev_clean, cmap='viridis', s=50, edgecolor='k')





# Add colorbar
cbar = plt.colorbar(sc, ax=ax, orientation='vertical', label="Elevation (m)")

# Labels and title
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title("Station Locations with Elevation")
    
  
    
# Make a map plot
to_plot = metar_xr.isel(time=0)




    
    
        
    
    
    
    