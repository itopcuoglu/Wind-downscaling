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
import numpy as np
import glob
import matplotlib.cm as cm
import xarray as xr
from datetime import datetime, timedelta
import time
import os


#save_folder = "/kfs2/projects/sfcwinds/HRRR"
save_folder = "../data/HRRR"

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


def safe_xarray(herbie_obj, search_str):
    
    """ accessing the Herbie object sometimes gives a permission error, adding a wait time helps."""
    
    for _ in range(5):  # Retry up to 5 times
        try:
            return herbie_obj.xarray(search_str, remove_grib=True)
        except PermissionError:
            time.sleep(1)  # Wait and retry
    raise Exception(f"Failed to load {search_str} after multiple attempts.")





#%%

if __name__ == "__main__":

    
    # Define download area
    
    # Have not found a way to slice by lat and lon directly, use x,y instead (below)
    min_lat = 35
    max_lat = 38
    min_lon = -111
    max_lon = -118
    
    # Define area to download (boundaries of download area)
    area = "NM_AZ"   #   "NM_AZ", "US", "US_SW"
    
    
    if area == "NM_AZ":   # New Mexico and Arizona
        slicex = slice(300, 800)
        slicey = slice(300, 600)
    
    elif area == "US":  # Entire continental US
        slicex = slice(130, 1730)
        slicey = slice(50, 1040)       
    
    elif area == "US_SW":  # US Southwest
        slicex = slice(130, 1000)
        slicey = slice(50, 800)    
    
    # date range with one-day frequency
    date_range = pd.date_range(datetime(2020, 1, 1), 
                               datetime(2024, 12, 31), 
                               freq="D").tolist()[::-1]   # list starts from end

    # Forecast hour (0=analysis, 2 is recommended)
    fxx = 2    
    
    #%% read HRRR in time range
    
    

    
    # plt.figure()
    
    # loop over days in date_range
    for date in date_range[:1]: 
        
        print(date)
        
        daily_file_path = os.path.join(save_folder, f"hrrr_{area}_{date.date()}.nc")
        
        # Check if file already exists, exit loop if so
        if os.path.exists(daily_file_path):
            print(f"File already exists: {daily_file_path}.")
            continue
        
        daily_datasets = []  # Store hourly datasets for the day
    
        for hour in range(-1, 23, 1):  # Looping to cover 0h-22h, so valid_date goes up to 24h
        
            max_retries = 5  # Set a limit to avoid infinite loops
            attempt = 0
            
            while attempt < max_retries:
                
                try:
                
                    time = date + pd.to_timedelta(hour, "H") 
                    valid_time = date + pd.to_timedelta(hour + 2, "H")  # 2-hour shift
                    
                    #valid_date = date + pd.to_timedelta(2, "H")
                    
                    print (valid_time)
                    
                    # Define file path
                    file_path = os.path.join(save_folder, f"hrrr_{area}_{valid_time.date()}_{valid_time.hour:02d}h.nc")
                    
                    print("sfc file")
                
                    # get hourly surface data during this day
                    H_sfc = herbie.fast.FastHerbie(
                            [time], #add an hour to get the subsequent subhourly steps
                            prioriy = 'google',
                            model="hrrr",
                            product="sfc", # to get 15min steps backwards, set fxx to 1 and product "subh", otherwise "sfc", alternatively "nat"
                            fxx=[fxx]
                        )
                    
                    #H_sfc.file_exists   
                    #H_sfc.inventory().variable.values   
                    #H_sfc.inventory()[["variable", "search_this"]].values    
                    # H.xarray("(?:HGT|LAND):surface")
                           
                    # download seperate xarrays for different search strings (why cannot search with one search string? - maybe different steps for each variable?)   
                    df = H_sfc.inventory()[["variable", "search_this"]]
                    filtered_df = df[df["search_this"].str.contains(r"\b(?:ugrd|vgrd)\b", case=False, na=False) & 
                                     df["search_this"].str.contains("mb", case=False, na=False)]
                    wind_search_string = "|".join(filtered_df["search_this"].drop_duplicates().values)
                    
                    ds1 = safe_xarray(H_sfc, wind_search_string)
                    ds2 = safe_xarray(H_sfc, ':TMP:2 m')
                    ds3 = safe_xarray(H_sfc, 'GUST|:PRES:surface:2 hour fcst|:SNOWC:surface:2 hour fcst|:APCP:surface:0-2 hour acc fcst|:SFCR:surface:2 hour fcst|:VEG:surface:2 hour fcst|:VGTYP:surface:2 hour fcst|:HPBL:surface:2 hour fcst')
                    ds4 = safe_xarray(H_sfc, '[U\|V]GRD:10 m')
                    ds5 = safe_xarray(H_sfc, '[U\|V]GRD:80 m')
                    ds5 = ds5.rename({"u": "u80",
                                            "v": "v80",                              
                                            })
                    ds6 = safe_xarray(H_sfc, ':DPT:2 m above ground:2 hour fcst')
                    ds7 = safe_xarray(H_sfc, ':TCDC:boundary layer cloud layer:2 hour fcst')
                    ds8 = safe_xarray(H_sfc, ':CAPE:0-3000 m above ground:2 hour fcst')
                    
            
                    ds1 = ds1.sel(y=slicey, x=slicex)  
                    ds2 = ds2.sel(y=slicey, x=slicex)  
                    ds3 = ds3.sel(y=slicey, x=slicex) 
                    ds4 = ds4.sel(y=slicey, x=slicex)
                    ds5 = ds5.sel(y=slicey, x=slicex)
                    ds6 = ds6.sel(y=slicey, x=slicex) 
                    ds7 = ds7.sel(y=slicey, x=slicex)
                    ds8 = ds8.sel(y=slicey, x=slicex)            
         
                    
                    ds_sfc = xr.merge([ds1,ds2, ds3, ds4, ds5, ds6, ds7, ds8],compat='minimal')
                    
                    # ds_sfc['wspd10'] = (ds_sfc.u10**2 + ds_sfc.v10**2)**0.5
                    # ds_sfc['wdir10'] = np.degrees(np.arctan2(ds_sfc.u10, ds_sfc.v10)) +180
                    
                    
                    ds_sfc = ds_sfc.rename({"u10": "u10_h",
                                            "v10": "v10_h",
                                            # "wspd10": "wspd10_h",
                                            # "wdir10": "wdir10_h"                              
                                            })
                    
                    # Add the time dimension
                    ds_sfc = ds_sfc.expand_dims(dim="valid_time")
                    
                    
                    print("subh file")
                    
                    # get subhourly data during this day
                    H_subh = herbie.fast.FastHerbie(
                            [time],
                            prioriy = 'google',
                            model="hrrr",
                            product="subh", # to get 15min steps backwards, set fxx to 1 and product "subh", otherwise "sfc", alternatively "nat"
                            fxx=[fxx]
                        )
                    
                    
                    #H_subh.inventory()[["variable", "search_this"]].values 
                    
                    # Extract wind variables
                    df = H_subh.inventory()[["variable", "search_this"]]
                    filtered_df = df[df["search_this"].str.contains(r"\b(?:ugrd|vgrd)\b", case=False, na=False) & 
                                     df["search_this"].str.contains("ave", case=False, na=False)]
                    wind_search_string = "|".join(filtered_df["search_this"].drop_duplicates().values)
                    
               
                    ds1 = safe_xarray(H_subh, wind_search_string)#.isel(step=[0,1,2]) # skip the last step because this is the forecast for the next hour
                    ds2 = safe_xarray(H_subh, ':TMP:2 m|DPT')#.isel(step=[0,1,2])
                    ds3 = safe_xarray(H_subh, 'GUST')#.isel(step=[0,1,2])
                    
                    ds1 = ds1.sel(y=slicey, x=slicex)   
                    ds2 = ds2.sel(y=slicey, x=slicex)  
                    ds3 = ds3.sel(y=slicey, x=slicex)   
                    
                    ds_subh = xr.merge([ds1,ds2, ds3],compat='minimal')
                    
                    # ds_subh['wspd10'] = (ds_subh.u10**2 + ds_subh.v10**2)**0.5
                    # ds_subh['wdir10'] = np.degrees(np.arctan2(ds_subh.u10, ds_subh.v10)) +180
                
                    ds_subh = ds_subh.swap_dims({"step": "valid_time"})   
                    
                        
                    # # get native data during this day - probably not needed for surface wind analysis
                    # H_nat = herbie.fast.FastHerbie(
                    #         time_range[:],
                    #         prioriy = 'google',
                    #         model="hrrr",
                    #         product="nat", # to get 15min steps backwards, set fxx to 1 and product "subh", otherwise "sfc", alternatively "nat"
                    #         fxx=[2]
                    #     )
                    
                    # #H_nat.inventory()[["variable", "search_this"]].values 
                           
                    # # download seperate xarrays for different search strings 
                    # ds1 = safe_xarray(H_nat, ':UGRD:1 hybrid level:2 hour fcst|:VGRD:1 hybrid level:2 hour fcst')
                    # ds1 = ds1.sel(y=slicey, x=slicex)  
                    
                    # ds_nat = xr.merge([ds1],compat='minimal')
                
                    # # Add the time dimension
                    # ds_nat = ds_nat.expand_dims(dim="valid_time")
                
                    print("merge")
              
                    # Merge hourly surface files and subhourly files
                    hrrr = xr.merge([ds_sfc, ds_subh], compat='override')  # storage space is less when saving in combines datafile
                
                    
                    # Reduce file size (drop vars, chunking, compression)
                    hrrr = hrrr.drop_vars(["boundaryLayerCloudLayer", "heightAboveGroundLayer", "surface", "gribfile_projection", "time"], errors="ignore")
                    hrrr.attrs = {}
                    for var in hrrr.data_vars:
                        hrrr[var].attrs = {}
                    
                    comp = {"zlib": True, "complevel": 9}  # 1 being fastest, but lowest compression ratio, 9 being slowest but best compression ratio  
                    encoding = {
                                var: {**comp, "chunksizes": (4, 2, 100, 100)} # 
                                if len(hrrr[var].dims) == 4 else {**comp, "chunksizes": (4, 100, 100)}
                                for var in hrrr.data_vars
                            }
                    
                    # if 'isobaricInhPa' in hrrr.dims:
                    #     hrrr = hrrr.chunk({"valid_time": 4, "isobaricInhPa": 2, "y": 100, "x": 100})
                    # else:
                    #     hrrr = hrrr.chunk({"valid_time": 4,  "y": 100, "x": 100})
                        
                    # Store dataset in list
                    daily_datasets.append(hrrr)
                        
                    # hrrr.to_netcdf(file_path, encoding=encoding) # '../data/HRRR\\hrrr_test.nc'
                    # # load: hrrr = xr.open_dataset(save_folder+"/hrrr_NM_AZ_2024-12-31_00h.nc", chunks="auto")
                    # print ("Saved hourly file.")
                        
                except:
                    attempt += 1
                    print(f"Attempt {attempt} failed. Retrying...")
                    time.sleep(1)  # Wait before retrying

            
            
                
        # Merge all hourly datasets into a daily file
        daily_hrrr = xr.concat(daily_datasets, dim="valid_time")  
            
        # save daily file
        daily_hrrr.to_netcdf(daily_file_path, encoding=encoding)
           
        # file_pattern = os.path.join(save_folder, f"hrrr_{area}_{valid_time.date()}_*h.nc")
        # file_list = sorted(glob.glob(file_pattern))
        # if file_list:
        #     hrrr = xr.open_mfdataset(file_list[:], concat_dim ="valid_time",combine='nested', chunks="auto", parallel = True)

   
        # test hourly vs subourly wind speed
        plt.figure()
        plt.plot(daily_hrrr.valid_time.values, daily_hrrr.isel(x=100, y=100).u10_h.values, ".", color="blue", label = "HRRR sfc")    
        plt.plot(daily_hrrr.valid_time.values, daily_hrrr.isel(x=100, y=100).u10.values, ".", color="orange", label = "HRRR subh")    
                        
#%% H5 files              
                
# """ this is a test with h5 files to reduce file size - does not help much."""
# make_h5_file = False
# if make_h5_file == True:
    
#     import h5py

#     # Replace u and v at pressure levels - does not save disk space
#     hrrr['u_500hPa'] = hrrr['u'].sel(isobaricInhPa=500.0)  # Selecting u at 500 hPa
#     hrrr['v_500hPa'] = hrrr['v'].sel(isobaricInhPa=500.0)  # Selecting v at 500 hPa
    
#     hrrr['u_300hPa'] = hrrr['u'].sel(isobaricInhPa=300.0)  # Selecting u at 300 hPa
#     hrrr['v_300hPa'] = hrrr['v'].sel(isobaricInhPa=300.0)  # Selecting v at 300 hPa
    
#     hrrr['u_250hPa'] = hrrr['u'].sel(isobaricInhPa=250.0)  # Selecting u at 300 hPa
#     hrrr['v_250hPa'] = hrrr['v'].sel(isobaricInhPa=250.0)  # Selecting v at 300 hPa
    
    
#     # Step 3: Drop the original 'u' and 'v' variables
#     hrrr = hrrr.drop_vars(['u', 'v'])
#     hrrr = hrrr.drop_vars(['isobaricInhPa'])
    
    
#     #### Reduce netcdf to valid_time - coordinates (coordinates = time * (x*y)) - this does not save any storage space in netcdf either,
#     ds = hrrr
    
#     # Get original spatial shape
#     y_size, x_size = ds.sizes['y'], ds.sizes['x']
#     num_points = y_size * x_size  # Total number of spatial points
    
#     # Flatten latitude and longitude, then stack into a (2, x*y) shape
#     lat_flat = ds['latitude'].values.ravel()  # Flatten to (x*y,)
#     lon_flat = ds['longitude'].values.ravel()  # Flatten to (x*y,)
#     coordinates = np.vstack([lat_flat, lon_flat])  # Shape (2, x*y)
    
#     # Reshape all spatial variables to match new dimensions (valid_time, num_points)
#     new_data_vars = {}
#     for var_name, var_data in ds.data_vars.items():
#         if 'y' in var_data.dims and 'x' in var_data.dims:
#             # Flatten spatial dimensions while keeping valid_time
#             new_data_vars[var_name] = (('valid_time', 'coordinates'), var_data.values.reshape(len(ds['valid_time']), num_points))
#         elif 'valid_time' in var_data.dims:
#             # Keep variables that do not depend on (y, x) unchanged
#             new_data_vars[var_name] = var_data
    
#     # Create a new xarray dataset with modified dimensions
#     new_ds = xr.Dataset(
#         data_vars=new_data_vars,
#         coords={
#             'valid_time': ds['valid_time'],  # Keep valid_time
#             'coordinates': (('dim', 'coordinates'), coordinates)  # New 2-row coordinate variable
#         }
#     )


#     # Create a new HDF5 file to save the data
#     h5_file = '../data/HRRR\\hrrr_NM_AZ_2024-12-31_00h_test.h5'   # file_path
    
#     with h5py.File(h5_file, 'w') as h5f:
  
#         # Save "coordinates" dataset (shape: 2, x*y) with compression
#         h5f.create_dataset('coordinates', data=new_ds['coordinates'].values, 
#                             compression='gzip', compression_opts=9)
        
#         # Save "valid_time" (convert datetime to string format for HDF5 compatibility)
#         valid_time_str = new_ds['valid_time'].astype(str).values  # Convert to string
#         h5f.create_dataset('valid_time', data=valid_time_str.astype('S'), 
#                             compression='gzip', compression_opts=9)  # Store as bytes
        
#         # Save all transformed data variables (valid_time, coordinates) with compression
#         for var_name, var_data in new_ds.data_vars.items():
#             h5f.create_dataset(var_name, data=var_data.values, 
#                                compression='gzip', compression_opts=9)
        


#     # # read the h5 file
#     # with h5py.File(h5_file, 'r') as h5f:

        
#     #     # Print the names of the datasets in the file
#     #     print(list(h5f.keys()))
       
#     #     dataset = h5f['d2m']
#     #     print(dataset.shape)
#     #     print(dataset[:10])
    
                

        
#%% Plots
    
    
    
    # # Make a map plot
    # to_plot = ds_sfc.isel(valid_time=0).isel()
    
    # fig, ax = plt.subplots(figsize=(10, 6), subplot_kw={'projection': ccrs.PlateCarree()})
    # plt.pcolormesh(to_plot.longitude, to_plot.latitude, to_plot.gust, cmap="viridis", shading="auto", transform=ccrs.PlateCarree())
    # plt.colorbar(label="Wind Speed (m/s)")
    
    # # Add map features
    # ax.add_feature(cfeature.COASTLINE)
    # ax.add_feature(cfeature.BORDERS, linestyle=':')
    # ax.add_feature(cfeature.STATES, linestyle='-', linewidth=0.5, edgecolor="gray") 
    # ax.add_feature(cfeature.LAND, edgecolor='black', alpha=0.3)
    # ax.add_feature(cfeature.LAKES, edgecolor='black', alpha=0.3)
    # ax.add_feature(cfeature.RIVERS, edgecolor='blue', alpha=0.3)
    # terrain = cfeature.NaturalEarthFeature(
    #     category='physical',
    #     name='shaded_relief',
    #     scale='110m'  # High resolution
    # )
    
    
    
    
    
    
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
        