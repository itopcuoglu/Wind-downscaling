from herbie import Herbie
import herbie
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import shapely.geometry as sgeom
import cartopy.feature
import cartopy.feature as cfeature
from cartopy.feature import ShapelyFeature
from cartopy.io.shapereader import Reader
import numpy as np
import pandas as pd
import numpy as np
import glob
import matplotlib.cm as cm
import xarray as xr
from datetime import datetime, timedelta
import time
import os
import sys
import warnings
warnings.filterwarnings("ignore", message="Will not remove GRIB file because it previously existed.")
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)

import dask
from dask.distributed import Client, LocalCluster
from dask import delayed, compute

def flush_print(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()
    sys.stderr.flush()






"""

HRRR model documentation:
https://rapidrefresh.noaa.gov/hrrr/

Documentation for the HRRR data download wrapper:
https://herbie.readthedocs.io/en/stable/

Variables:
https://www.nco.ncep.noaa.gov/pmb/products/hrrr/hrrr.t00z.wrfnatf00.grib2.shtml


"""


#%% Define

# Folder to save datasets
save_folder = "/kfs2/projects/sfcwinds/HRRR"
# save_folder = "../data/HRRR"


# Define area to download (boundaries of download area) 
# Have not found a way to slice by lat and lon directly, use x,y instead (below)
area = "US_SW"   #   "NM_AZ", "US", "US_SW"

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
date_range = pd.date_range(datetime(2014, 1, 1), 
                           datetime(2016, 8, 22), 
                           freq="D").tolist()[::-1]   # list starts from end

# Forecast hour (0=analysis, 2 is recommended)
fxx = 2    

# Chunk dictionaries
chunk_dict_hourly = {"valid_time": 1, "isobaricInhPa": 1, "y": 200, "x": 200}
chunk_dict_daily = {"valid_time": 12, "isobaricInhPa": 1, "y": 200, "x": 200}




# Define compression settings
comp = {"zlib": True, "complevel": 4}  


#%% Functions


def safe_xarray(herbie_obj, search_str):
    
    """ accessing the Herbie object sometimes gives a permission error, adding a wait time helps."""
    
    for _ in range(5):  # Retry up to 5 times
        try:
            return herbie_obj.xarray(search_str, remove_grib=True)
        except PermissionError:
            time.sleep(1)  # Wait and retry
    raise Exception(f"Failed to load {search_str} after multiple attempts.")
    
    
def process_hourly_data(timestamp, area, slicey, slicex, fxx):
    """Retrieve, process, and return hourly HRRR dataset"""
    
    max_retries = 5  # Set a limit to avoid infinite loops
    attempt = 0
    
    while attempt < max_retries:
        
        try:
            valid_time = timestamp + pd.to_timedelta(fxx, "h") # 2-hour shift (forecast lead time fxx)
    
            flush_print(f"{valid_time}")

    
            H_sfc = Herbie(timestamp, model='hrrr', product='sfc', fxx=fxx, verbose=False)
            
            if timestamp > pd.to_datetime('2018-09-16 00:00:00'):   # this changed on the given date, also only forecast step 0 available in all subhourly files before this date
                H_subh = Herbie(timestamp, model='hrrr', product='subh', fxx=fxx, verbose=False)
            else:
                H_subh = Herbie(timestamp, model='hrrr', product='subh', fxx_subh=fxx, verbose=False)

            
            #H_sfc.inventory().variable.values   
            #H_sfc.inventory()[["variable", "search_this"]].values 
    
            # Define search patterns for variables
            # df = H_sfc.inventory()[["variable", "search_this"]]
            # filtered_df = df[df["search_this"].str.contains(r"\b(?:ugrd|vgrd)\b", case=False, na=False) & 
            #                  df["search_this"].str.contains("mb", case=False, na=False)]
            # wind_search_string_sfc = "|".join(filtered_df["search_this"].drop_duplicates().values)
            wind_search_string_sfc = ':UGRD:250 mb:2 hour fcst|:VGRD:250 mb:2 hour fcst|:UGRD:500 mb:2 hour fcst|:VGRD:500 mb:2 hour fcst|:UGRD:850 mb:2 hour fcst|:VGRD:850 mb:2 hour fcst'

            
            
            df = H_subh.inventory()[["variable", "search_this"]]
            if timestamp > pd.to_datetime('2016-08-21 00:00:00'):   # search string name changed on the given date
                filtered_df = df[df["search_this"].str.contains(r"\b(?:ugrd|vgrd)\b", case=False, na=False) & 
                                 df["search_this"].str.contains("ave", case=False, na=False)]
            else:
                filtered_df = df[df["search_this"].str.contains(r"\b(?:ugrd|vgrd):10\b", case=False, na=False)]                
            wind_search_string_subh = "|".join(filtered_df["search_this"].drop_duplicates().values)
                  
    
            
            ### Surface data
            ds_sfc = xr.merge([
                safe_xarray(H_sfc, wind_search_string_sfc),
                safe_xarray(H_sfc, ':TMP:2 m'),
                safe_xarray(H_sfc, 'GUST|:PRES:surface:2 hour fcst|:SNOWC:surface:2 hour fcst|:APCP:surface:0-2 hour acc fcst|:SFCR:surface:2 hour fcst|:VEG:surface:2 hour fcst|:VGTYP:surface:2 hour fcst|:HPBL:surface:2 hour fcst'),
                safe_xarray(H_sfc, '[U|V]GRD:10 m'),
                safe_xarray(H_sfc, '[U|V]GRD:80 m').rename({"u": "u80", "v": "v80"}),
                safe_xarray(H_sfc, ':DPT:2 m above ground:2 hour fcst'),
            ], compat='minimal')   # needs to be minimal to keep the isobaricInhPa dimension

            # In cases when u and v have only one value, isobaricInhPa is not a dimension
            if "isobaricInhPa" not in ds_sfc.dims:
                ds_sfc = ds_sfc.assign(
                    u=ds_sfc.u.expand_dims(isobaricInhPa=[ds_sfc.isobaricInhPa.values]),
                    v=ds_sfc.v.expand_dims(isobaricInhPa=[ds_sfc.isobaricInhPa.values])
                )  

            # Add step as dimension when only one step in the data (then it's not a dimension)
            if "step" not in ds_sfc.dims:
                ds_sfc = (
                    ds_sfc
                    .expand_dims({"step": ds_sfc.step.values[np.newaxis]})
                    .assign_coords(
                        step=("step", ds_sfc.step.values[np.newaxis]),
                        valid_time=("step", ds_sfc.valid_time.values[np.newaxis])
                    )
                )
            
            # Use only the fxx forecast step (and 45min back for subh)
            ds_sfc = ds_sfc.sel(
                step=(ds_sfc.step <= pd.Timedelta(f"{fxx}h")) &
                      (ds_sfc.step > pd.Timedelta(f"{fxx-1}h"))
            )

            # Slice spatially and use the valid_time dimension instead step
            ds_sfc = ds_sfc.sel(y=slicey, x=slicex).swap_dims({"step": "valid_time"})
            
            # ds_sfc['wspd10'] = (ds_sfc.u10**2 + ds_sfc.v10**2)**0.5
            # ds_sfc['wdir10'] = np.degrees(np.arctan2(ds_sfc.u10, ds_sfc.v10)) +180
            
            ds_sfc = ds_sfc.rename({"u10": "u10_h",
                                    "v10": "v10_h",
                                    "gust": "gust_h",
                                    # "wspd10": "wspd10_h",
                                    # "wdir10": "wdir10_h"                              
                                    })
        
    
    
            ### Subhourly data
            ds_subh = xr.merge([
                safe_xarray(H_subh, wind_search_string_subh),
                # safe_xarray(H_subh, ':TMP:2 m|DPT')#.isel(step=[0,1,2])
                safe_xarray(H_subh, 'GUST')
            ], compat='override')   # needs to be override to keep the valid_time dimension

            # Add step as dimension when only one step in the data (then it's not a dimension)
            if "step" not in ds_subh.dims:
                ds_subh = (
                    ds_subh
                    .expand_dims({"step": ds_subh.step.values[np.newaxis]})
                    .assign_coords(
                        step=("step", ds_subh.step.values[np.newaxis]),
                        valid_time=("step", ds_subh.valid_time.values[np.newaxis])
                    )
                )

            
            # Use only the fxx forecast step (and 45min back for subh)
            ds_subh = ds_subh.sel(
                step=(ds_subh.step <= pd.Timedelta(f"{fxx}h")) &
                      (ds_subh.step > pd.Timedelta(f"{fxx-1}h"))
            )

            # Slice spatially and use the valid_time dimension instead step
            ds_subh = ds_subh.sel(y=slicey, x=slicex).swap_dims({"step": "valid_time"})
                               
            try:                   
                ds_subh = ds_subh.rename({"avg_10u": "u10",
                                    "avg_10v": "v10",                            
                                    })  
            except:
                pass
    
            # Merge hourly and subhourly datasets
            hrrr = xr.merge([ds_sfc, ds_subh])
    
            # Cleanup variables and attributes
            hrrr = hrrr.drop_vars(["boundaryLayerCloudLayer", "heightAboveGroundLayer", "heightAboveGround",
                                   "surface", "gribfile_projection", "time"], errors="ignore")

    
            return hrrr.chunk(chunk_dict_hourly)  # Apply chunking to hourly files
        
        except Exception as e:
    
            flush_print(f"Error: {e}")
            
            attempt += 1
            flush_print(f"Attempt {attempt} failed. Retrying...")
            time.sleep(1)  # Wait before retrying        











if __name__ == "__main__":
    

    start_time = time.perf_counter()


    # Initialize Dask Client for parallel processing - does not work
    # cluster = LocalCluster(n_workers=16, threads_per_worker=1, memory_limit="15GB")  # Adjust the number of workers based on the CPUs per task in SLURM
    # client = Client(cluster)

    # cluster = LocalCluster(n_workers=1, threads_per_worker=1, memory_limit="240GB")
    # client = Client(cluster)
    # print(f"Dask dashboard available at: {client.dashboard_link}")

    
 
    # loop over days in date_range
    for date in date_range[:]: 
        
        flush_print(date.date())

        try:
            
            daily_file_path = os.path.join(save_folder, f"hrrr_{area}_{date.date()}.nc")
            
            # # Check if file already exists, exit loop if so
            # if os.path.exists(daily_file_path):
            #     flush_print(f"File already exists: {daily_file_path}.")
            #     continue

            daily_datasets  = []  # Store hourly datasets for the day

            delayed_datasets  = []  # for dask processing
            try:
                del daily_hrrr
            except:
                pass
        
            for hour in range(-fxx+1, fxx+21, 1):  # Looping to hours, so that valid_date goes from 0h to 24h
                
                timestamp = date + pd.to_timedelta(hour, "h")
                    
                # Load and process  hourly files 
                
                # Schedule processing as a Dask delayed computation
                delayed_hrrr = delayed(process_hourly_data)(timestamp, area, slicey, slicex, fxx)
                delayed_datasets.append(delayed_hrrr)

            # Compute all hourly datasets in parallel
            daily_datasets = list(compute(*delayed_datasets))

            # Filter out None in case of failures
            daily_datasets = [ds for ds in daily_datasets if ds is not None]

            if not daily_datasets:
                continue
        
            # Merge all hourly datasets into a daily file
            daily_hrrr = xr.concat(daily_datasets, dim="valid_time", combine_attrs="override") 
            daily_hrrr = daily_hrrr.chunk(chunk_dict_daily)

            # Sort by time
            daily_hrrr = daily_hrrr.sortby('valid_time')
            
                
            # save daily file
            encoding = {
                var: {**comp, "chunksizes": (12, 1, 200, 200)} # 
                if len(daily_hrrr[var].dims) == 4 else {**comp, "chunksizes": (1, 200, 200)}
                for var in daily_hrrr.data_vars
                }
            daily_hrrr.to_netcdf(daily_file_path, encoding=encoding)
            # print ("daily file saved "+daily_file_path)

        except:
            pass
        
        
           
        # # Load the files:
        # file_pattern = os.path.join(save_folder, f"hrrr_{area}_{valid_time.date()}_*h.nc")
        # file_list = sorted(glob.glob(file_pattern))
        # if file_list:
        #     hrrr = xr.open_mfdataset(file_list[:], concat_dim ="valid_time",combine='nested', chunks="auto", parallel = True)

   
        # test hourly vs subourly wind speed
        # plt.figure()
        # plt.plot(daily_hrrr.valid_time.values, daily_hrrr.isel(x=100, y=100).u10_h.values, ".", color="blue", label = "HRRR sfc")    
        # plt.plot(daily_hrrr.valid_time.values, daily_hrrr.isel(x=100, y=100).u10.values, ".", color="orange", label = "HRRR subh")    
                      
        
    # Close the Dask client after processing
    # client.close()    
    
    
    # Timing
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.4f} seconds")













        
#%% H5 files              
                
# """ this is a test with h5 files to reduce file size 
#     - does not help much.
#     - storage size is similar (h5 1.38GB vs nc 1.3GB for daily US file)
#     - speed?
#     """
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
    
#     hrrr['u_700hPa'] = hrrr['u'].sel(isobaricInhPa=700.0)  # Selecting u at 500 hPa
#     hrrr['v_700hPa'] = hrrr['v'].sel(isobaricInhPa=700.0)  # Selecting v at 500 hPa
    
#     hrrr['u_1000hPa'] = hrrr['u'].sel(isobaricInhPa=1000.0)  # Selecting u at 300 hPa
#     hrrr['v_1000hPa'] = hrrr['v'].sel(isobaricInhPa=1000.0)  # Selecting v at 300 hPa
    
#     hrrr['u_850hPa'] = hrrr['u'].sel(isobaricInhPa=850.0)  # Selecting u at 300 hPa
#     hrrr['v_850hPa'] = hrrr['v'].sel(isobaricInhPa=850.0)  # Selecting v at 300 hPa
    
#     hrrr['u_925hPa'] = hrrr['u'].sel(isobaricInhPa=925.0)  # Selecting u at 300 hPa
#     hrrr['v_925hPa'] = hrrr['v'].sel(isobaricInhPa=925.0)  # Selecting v at 300 hPa
    
    
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
#     h5_file = '../data/HRRR\\test.h5'   # file_path
    
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
#                                 compression='gzip', compression_opts=9)
        


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
        