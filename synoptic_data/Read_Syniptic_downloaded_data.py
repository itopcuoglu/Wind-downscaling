# -*- coding: utf-8 -*-
"""
Created on Fri Apr  4 18:35:13 2025

@author: uegerer
"""

import json
import pandas as pd
import glob
import os




# Initialize an empty list to store DataFrames
combined_data = []

# Loop through the  JSON files in the "synoptic_data/" folder
json_folder = "synoptic_data/"
json_files = [f for f in os.listdir(json_folder) if f.endswith(".json")]



for filename in json_files[:]:
    if filename.endswith(".json"):
        
        print(filename)
        
        # Read the JSON file
        with open(os.path.join(json_folder, filename), "r") as f:
            data = json.load(f)
        
        # Extract the station metadata and observation data
        station = data["STATION"][0]  # Assuming one station per file
        observations = station.get("OBSERVATIONS", {})
        
        # Extract metadata
        stid = station["STID"]
        name = station["NAME"]
        elevation = station["ELEVATION"]
        latitude = station["LATITUDE"]
        longitude = station["LONGITUDE"]
        state = station["STATE"]
        
        # Extract the position of the wind sensor (wind_speed_set_1)
        wind_speed_position = station["SENSOR_VARIABLES"]["wind_speed"]["wind_speed_set_1"].get("position")
        
        # Create a DataFrame from the observations
        df = pd.DataFrame(observations)
        
        # Remove '_set_1' from the column names
        df.columns = df.columns.str.replace('_set_1', '', regex=False)
        
        df.rename(columns={
            'air_temp': 'temperature',
            'wind_speed': 'windspeed',
            'wind_direction': 'winddirection',
            'wind_gust': 'gust'
        }, inplace=True)
        
        # Convert 'date_time' to datetime and set as index
        df['date_time'] = pd.to_datetime(df['date_time'])
        df.set_index('date_time', inplace=True)
        
        # Add the station metadata as new columns
        df['stid'] = stid
        df['station_name'] = name
        df['elev'] = elevation
        df['lat'] = latitude
        df['lon'] = longitude
        df['state'] = state
        df['height'] = wind_speed_position
        
        df['elev'] = pd.to_numeric(df['elev'], errors='coerce') * 0.3048   # conversion from ft to m
        
        # Append the DataFrame to the list
        combined_data.append(df)

# Combine all DataFrames into a single DataFrame
final_df = pd.concat(combined_data, ignore_index=False)



# Save the combined DataFrame to a CSV or Excel file
final_df.to_pickle('obs_NM_synoptic_2024.pkl')
























# # Define the path to the subfolder
# folder_path = 'synoptic_data/'

# # Use glob to find all JSON files in the folder (and its subfolders)
# json_files = glob.glob(os.path.join(folder_path, '**', '*.json'), recursive=True)

# # Print the list of JSON files found
# for file in json_files:
#     print(file)






# # Load the JSON data from the file
# with open(file, 'r') as f:
#     data = json.load(f)

# # Extract the observations from the data
# observations = data["STATION"][0]["OBSERVATIONS"]

# # Convert the observation times and any other variables into a DataFrame
# df = pd.DataFrame(observations)

# # Remove '_set_1' from the column names
# df.columns = df.columns.str.replace('_set_1', '', regex=False)

# # Convert 'date_time' to datetime and set it as the index
# df['date_time'] = pd.to_datetime(df['date_time'])
# df.set_index('date_time', inplace=True)


