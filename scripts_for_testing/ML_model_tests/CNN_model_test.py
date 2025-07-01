# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 12:44:14 2025

@author: uegerer
"""

# !pip install tensorflow


import tensorflow as tf
# from tensorflow.keras import layers, models

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


Sandia_coords = [34.96226779999999, -106.5096873 + 360]



#%% Define the 3D CNN Model


"""
CNN Architecture Design

A simple 3D CNN architecture could look like this:

    Input Layer: Your input will be a 4D tensor with shape (batch_size, time_steps, lat, lon, features).

    Conv3D Layers:
        Apply 3D convolutions to learn spatial and temporal patterns.
        A good starting point would be:
            Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu'): 3D filter over time, latitude, and longitude.
            Use batch normalization after convolutions to stabilize learning.
            Add dropout after each layer to prevent overfitting (optional).

    MaxPooling3D: Apply 3D pooling to reduce the spatial and temporal dimensions.
        Example: MaxPooling3D(pool_size=(2, 2, 2)) reduces the dimensions in all three axes.

    Flatten Layer: After the convolutions, flatten the output to a 1D vector, ready for fully connected layers.

    Dense Layers: After flattening, apply fully connected layers:
        Example: Dense(64, activation='relu')
        Optionally, you can add more layers or dropout to regularize the network.

    Output Layer: If you're predicting a time series, the output shape might be (batch_size, num_predictions) or a grid of predictions for each lat-lon location.



Things to Consider

    Data Preprocessing:
        Normalize your input data (e.g., wind speed values, temperature) before feeding it into the network.
        Consider downsampling your input data if it’s too large (in terms of lat-lon grid size or time steps).

    Model Tuning:
        Experiment with different numbers of layers and filters (32, 64, 128, etc.) depending on the complexity of your problem.
        Adjust kernel sizes, pooling sizes, and dropout rates based on model performance.

    Evaluation:
        Monitor validation loss and MAE during training to avoid overfitting and make sure your model generalizes well to unseen data.


"""
    

def create_3d_cnn(input_shape):
    model = models.Sequential()

    # 3D Convolutional Layer
    model.add(layers.Conv3D(32, kernel_size=(3, 3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling3D(pool_size=(2, 2, 2)))

    # Second 3D Convolutional Layer
    model.add(layers.Conv3D(64, kernel_size=(3, 3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling3D(pool_size=(2, 2, 2)))

    # Third 3D Convolutional Layer
    model.add(layers.Conv3D(128, kernel_size=(3, 3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling3D(pool_size=(2, 2, 2)))

    # Flatten the 3D output into 1D for Dense layers
    model.add(layers.Flatten())

    # Fully Connected Layers
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))  # Dropout for regularization
    model.add(layers.Dense(64, activation='relu'))

    # Output Layer (e.g., predicting wind speed at new lat-lon points)
    model.add(layers.Dense(1))  # Output a single continuous value per sample (wind speed)

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

    return model



def find_nearest_obs(metar, target_lat = Sandia_coords[0], target_lon = Sandia_coords[1] ):

    return # station_lat, station_lon



#%% Read data


# Specify the date
year = "2021"
month = "01"
day = "*"



# HRRR data
file_pattern = f"../data/HRRR/hrrr_{year}-{month}-{day}_*h.nc"
file_list = sorted(glob.glob(file_pattern))
if file_list:
    hrrr = xr.open_mfdataset(file_list[:], combine="by_coords")
else:
    print("No HRRR files found for the specified date.")



# Metar data
metar = xr.open_dataset(f"../data/Metar/Metar_NM_ASOS_2021-01-01_2021-01-30.nc")
    

# Frequency of Metar messages
plt.figure()
plt.hist(np.diff(metar.time.values).astype('timedelta64[s]').astype(float)/60, bins = 100, range=(0,100))
for station in  np.unique(metar.station.values)[1:]:
    metar_plot = metar.where(metar.station==station,drop=True).wspd  
    plt.hist(np.diff(metar_plot.time.values).astype('timedelta64[s]').astype(float)/60, bins = 100, range=(0,100),label = station)
plt.legend()



# Define the max gap allowed for interpolation (in seconds)
max_gap = 20*60  

# Create a new dataset to store interpolated values
metar_interp = metar.copy()

# Iterate over each station
for station in  np.unique(metar.station.values)[1:]:
        # Extract time series for this location
        station_data = metar.where(metar.station==station)

        # Find valid time indices (non-NaN station values)
        valid_times = station_data.station.dropna('time').time

        if len(valid_times) < 2:
            continue  # Skip if there are not enough valid points for interpolation

        # Compute time differences between valid times
        time_diffs = np.diff(valid_times.values).astype('timedelta64[s]').astype(int)

        # Identify times where gaps are within the allowed range
        valid_interp_times = valid_times.values[:-1][time_diffs <= max_gap]
        valid_interp_times = np.append(valid_interp_times, valid_times.values[-1])

        # Perform interpolation only on allowed times
        metar_interp.where(metar_interp.station==station) = station_data.interp(time=valid_interp_times)



#%% Map Plot

day_to_plot = "2021-01-10T00:00:00"

hrrr_plot = hrrr.sel(time=day_to_plot, method = "nearest").wspd10
metar_time =  metar.sel(time = hrrr_plot.time, method='nearest').time.values
metar_plot = metar.sel(time =metar_time)['wspd'].T.values.flatten()



# Create a figure with a map projection
fig, ax = plt.subplots(figsize=(10, 6), subplot_kw={'projection': ccrs.PlateCarree()})
plt.title(f"HRRR day: {hrrr_plot.time.values} \n Metar day: {metar_time}")

# Add map features
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.add_feature(cfeature.STATES, linestyle='-', linewidth=0.5, edgecolor="gray") 
ax.add_feature(cfeature.LAND, edgecolor='black', alpha=0.3)
ax.add_feature(cfeature.LAKES, edgecolor='black', alpha=0.3)
ax.add_feature(cfeature.RIVERS, edgecolor='blue', alpha=0.3)

gl = ax.gridlines(draw_labels=True, linestyle="-", linewidth=0.5, alpha=0.0)
gl.right_labels = False  # Hide right labels
gl.top_labels = False  # Hide top labels

ax.set_extent([-110,-102, 31.5, 38], crs=ccrs.PlateCarree())  

ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")





# HRRR data
contour = plt.pcolormesh(hrrr_plot.longitude, 
                         hrrr_plot.latitude, 
                         hrrr_plot.squeeze(), cmap="viridis", shading="auto", transform=ccrs.PlateCarree(), vmin = 0, vmax = 15)
plt.colorbar(contour, ax=ax, orientation="vertical", label="Wind speed (m/s)")



# MEtar data
# Flatten the data and remove NaN values
lon, lat = np.meshgrid(metar['lon'], metar['lat'])
lon_flat = lon.flatten()
lat_flat = lat.flatten()
wind_flat = metar_plot

# Mask out NaN values
mask = ~np.isnan(wind_flat)
lon_clean = lon_flat[mask]
lat_clean = lat_flat[mask]
elev_clean = wind_flat[mask]

sc = plt.scatter(lon_clean, lat_clean, c=elev_clean, cmap='viridis', s=50, edgecolor='k', vmin = 0, vmax = 15)

# Sandia
ax.plot(Sandia_coords[1], Sandia_coords[0],transform=ccrs.PlateCarree(), marker='o', ms=10, color='black',  alpha=0.9)





#%% Time series Plot




# Slect MEtar grid point
metar_plot = metar.where(metar.station=="ABQ",drop=True).wspd   # Find function how to find station closest to target coordinates instead of selecting station manually!


# Select HRRR grid point (choose if original coordinate of interste or closest measurement station)
hrrr_coords_to_use = Sandia_coords
hrrr_coords_to_use = np.append(metar_plot.lat.values, metar_plot.lon.values+360)

dist = (hrrr['longitude'].values - hrrr_coords_to_use[1])**2 + (hrrr['latitude'].values - hrrr_coords_to_use[0])**2
idx = np.unravel_index(np.argmin(dist), dist.shape)
closest_lat = hrrr.latitude.values[idx]
closest_lon = hrrr.longitude.values[idx]

hrrr_plot =hrrr.sel(x=idx[1], y=idx[0]).wspd10



# Create a time series plot
fig, ax = plt.subplots(figsize=(10, 6))
plt.plot(metar_plot.time.values, metar_plot.values.flatten(),"-", label = "Metar", lw=0.5)
plt.plot(hrrr_plot.time.values, hrrr_plot.values.flatten(), "-", label = "HRRR", lw=0.5)
for station in  np.unique(metar.station.values)[1:]:
    metar_plot = metar.where(metar.station==station,drop=True).wspd   # Find function how to find station closest to target coordinates instead of selecting station manually!
    plt.plot(metar_plot.time.values, metar_plot.values.flatten(),".-", label = station, lw=0.5)
plt.legend()







#%% Train the model

# Example data shapes (you need to load your actual data)
input_shape = (30, 100, 100, 3)  # 30 time steps, 100x100 grid, 3 features (e.g., wind speeds)
# X_train = # Load your training data (shape: (num_samples, time_steps, lat, lon, features))
# y_train = # Load your target data (shape: (num_samples, 1)) - wind speed predictions

# # Create the model
# model = create_3d_cnn(input_shape)

# # Train the model
# model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)




# #%% Evaluation and prediction

# # Example test data
# X_test = # Load your test data
# y_test = # Load your test labels

# # Evaluate the model on test data
# loss, mae = model.evaluate(X_test, y_test)
# print(f"Test Loss: {loss}, Test MAE: {mae}")

# # Make predictions
# y_pred = model.predict(X_test)




#%% Save and load the model


# # Save the trained model
# model.save("wind_prediction_model.h5")

# # To load the model later
# loaded_model = tf.keras.models.load_model("wind_prediction_model.h5")


























