from herbie import Herbie
import herbie
#from toolbox import EasyMap, pc
#from paint.standard2 import cm_tmp
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import shapely.geometry as sgeom
import cartopy.feature
import cartopy.feature as cfeature
from cartopy.feature import ShapelyFeature
from cartopy.io.shapereader import Reader
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

HRRR model documentation:
https://rapidrefresh.noaa.gov/hrrr/

Documentation for the HRRR data download wrapper:
https://herbie.readthedocs.io/en/stable/

Variables:
https://www.nco.ncep.noaa.gov/pmb/products/hrrr/hrrr.t00z.wrfnatf00.grib2.shtml


"""


def merge_datasets(ds_list):
    """Merge list of Datasets together.

    Since cfgrib doesn't merge data in different "hypercubes", we will
    do the merge ourselves.

    Parameters
    ----------
    ds_list : list
        A list of xarray.Datasets, usually from the list of datasets
        returned by cfgrib when data is on multiple levels.
    """
    these = []
    for ds in ds_list:
        ds = ds.drop_vars("gribfile_projection")
        expand_dims = []
        for i in [
            "heightAboveGround",
            "time",
            "step",
            "isobaricInhPa",
            "depthBelowLandLayer",
        ]:
            if i in ds and i not in ds.dims:
                expand_dims.append(i)
        these.append(ds.expand_dims(expand_dims))
    return xr.merge(these, compat="override")




NSO = (-114.9753, 35.7970)
CrescentDunes = (-117.36360821139168, 38.23914286091489)





# Define download area

min_lat = 35
max_lat = 38
min_lon = -111
max_lon = -118


#%% read HRRR in time range


# date range with one-day frequency
date_range = pd.date_range(datetime(2021, 11, 1, 0, 0, 0), 
                           datetime(2021, 12, 31, 0, 0, 0), 
                           freq=timedelta(hours=1)).tolist()


# loop over days in date_range
for date in date_range: # [:1]:
    
    print (date)
    
    # Create an hourly range on this day
    # time_range = pd.date_range(date, 
    #                            date + pd.to_timedelta(1,"D"), 
    #                            freq=timedelta(hours=1)).tolist()[:-1]
    time_range = [date]
    


    # get (sub)hourly data during this day
    H_sfc = herbie.fast.FastHerbie(
            time_range[:],
            prioriy = 'google',
            model="hrrr",
            product="sfc", # to get 15min steps backwards, set fxx to 1 and product "subh", otherwise "sfc", alternatively "nat"
            fxx=[1]
        )
    
    #H_sfc.file_exists   
    #H_sfc.inventory().variable.values   
    #H_sfc.inventory()[["variable", "search_this"]].values    
    # H.xarray("(?:HGT|LAND):surface")
    # HPBL|SHTFL|S
           
    # download seperate xarrays for different search strings (why cannot search with one search string? - maybe different steps for each variable?)   
    ds1 = H_sfc.xarray('[U\|V]GRD:10 m')#.isel(step=[0,2,3,4])  
    ds2 = H_sfc.xarray(':TMP:2 m')
    ds3 = H_sfc.xarray('GUST')
    
    ds1 = ds1.sel(y=slice(100, 500), x=slice(100, 500))   # find way to slice with lat and lon!
    ds2 = ds2.sel(y=slice(100, 500), x=slice(100, 500))   # find way to slice with lat and lon!
    ds3 = ds3.sel(y=slice(100, 500), x=slice(100, 500))   # find way to slice with lat and lon!
    
    ds_sfc = xr.merge([ds1,ds2, ds3],compat='minimal')
    
    ds_sfc['wspd10'] = (ds_sfc.u10**2 + ds_sfc.v10**2)**0.5
    ds_sfc['wdir10'] = np.degrees(np.arctan2(ds_sfc.u10, ds_sfc.v10)) +180

    
    # Nearest point instead of area sclicing
    # dsi = ds.herbie.nearest_points(
    #     points=[NSO],
    #     names=["NSO"],
    # ).isel(point=0).isel(step=[0,2,3,4])   # why are there 2 values per step? for all besides U10 and v10 there are only 4 steps that work
            
        
    # get native data during this day
    H_nat = herbie.fast.FastHerbie(
            time_range[:2],
            prioriy = 'google',
            model="hrrr",
            product="nat", # to get 15min steps backwards, set fxx to 1 and product "subh", otherwise "sfc", alternatively "nat"
            fxx=[1]
        )
    
    #H_nat.inventory()[["variable", "search_this"]].values 
           
    # download seperate xarrays for different search strings 
    ds1 = H_nat.xarray(':UGRD:1 hybrid level:1 hour fcst|:VGRD:1 hybrid level:1 hour fcst')
    ds1 = ds1.sel(y=slice(100, 500), x=slice(100, 500))  
    
    ds_nat = xr.merge([ds1],compat='minimal')




    # Merge native resolution and surface files
    hrrr = merge_datasets([ds_sfc] + [ds_nat])

    # Rename the columns
    hrrr = hrrr.rename({
            'u': 'u_lev1',
            'v': 'v_lev1'
            })

    # # Add columns and interpolate 1h data
    # hrrr[["blh",  "ishf"]] = hrrr[["blh",  "ishf"]].interpolate()

    hrrr.to_netcdf(f"../data/HRRR/hrrr_{date.date()}_{date.hour:02d}h.nc")





# Make a map plot
to_plot = hrrr.isel(time=0).isel(step=0)

plt.figure()
plt.pcolormesh(to_plot.longitude, to_plot.latitude, to_plot.wspd10, cmap="viridis")
plt.colorbar(label="Wind Speed (m/s)")







# # Timeseries of wind
# fig = plt.figure(figsize=(11,4))
 
# ax2 = plt.subplot(2, 1, 1)
# plt.ylabel('Wind speed (m s$^{-1}$)')
# ax2.plot( hrrr.valid_time, hrrr.speed,'.', label = "HRRR, 10m", ms=3)
# ax2.plot( hrrr.valid_time, hrrr.gust,'.', label = "", ms=3, color="C2", alpha = 0.5)
# plt.legend(loc=1,  markerscale=2)
# plt.grid()
# ax2.set_zorder(100)
 
# ax1 = plt.subplot(2, 1, 2, sharex=ax2)  # 
# ax1.set_ylabel('Wind dir ($^\circ$)')
# hrrr['wdir'] = np.degrees(np.arctan2(hrrr.u10, hrrr.v10)) +180
# ax1.plot( hrrr.valid_time, hrrr.wdir,'.', label = "", ms=3)
# ax1.set_ylim(0, 360)    
# plt.grid()    
 
# #plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d %H:%M'))
# fig.autofmt_xdate()
# ax1.set_xlabel("Time (UTC)")
# plt.tight_layout()
# plt.subplots_adjust(hspace=0.1)
    