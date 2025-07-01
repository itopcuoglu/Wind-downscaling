# -*- coding: utf-8 -*-
"""
Created on Fri Apr  4 16:49:31 2025

@author: uegerer
"""

import requests
import json
import datetime
import pandas as pd


#%% Download observation data in a lopp for each station


# Define the API URL
url = "https://api.synopticdata.com/v2/stations/timeseries"

# List of stations
df = pd.read_csv("NM_stations_under_10m.csv")
station_ids = df.STID.values

for station in station_ids:    

    params = {
       #   "country": "us",
       #  "state": "nm",
        "stid": station,
       # "complete": "1",
        "sensorvars": "1",
        "start": "202404040000", # everything older than a year gets cut off.
        "end":   "202504040100",
       # "recent": "518400",  # in minutes
        "vars": "wind_speed,wind_direction,wind_gust,air_temp",
        "token": "55facdf6a0e74b8c84ea136072e06a66",
        "qc_checks " : "madis"
    }

    # Make the request
    response = requests.get(url, params=params, verify=False)

    # Check if successful
    if response.status_code == 200:
        data = response.json()
        # Save to file
        with open(f"synoptic_data/{station}_1year.json", "w") as f:
            json.dump(data, f, indent=2)
        # print("Data saved")
    else:
        print(f"Request failed with status code {response.status_code}")
        # print(response.text)
        

#%% This is more to get metadata


# t1 = datetime.datetime.now()

# # Define the API URL
# url = "https://api.synopticdata.com/v2/stations/timeseries"
# params = {
#    #   "country": "us",
#    #  "state": "nm",
#     "stid": "LSLN5",
#     "complete": "1",
#     "sensorvars": "1",
#     "start": "202404040000", # everything older than a year gets cut off.
#     "end":   "202504040100",
#    # "recent": "518400",  # in minutes
#     "vars": "wind_speed,wind_direction,wind_gust,air_temp",
#     "token": "55facdf6a0e74b8c84ea136072e06a66"
# }

# # Make the request
# response = requests.get(url, params=params, verify=False)

# # Check if successful
# if response.status_code == 200:
#     data = response.json()
#     # Save to file
#     with open("Synoptic_data.json", "w") as f:
#         json.dump(data, f, indent=2)
#     print("Data saved")
# else:
#     print(f"Request failed with status code {response.status_code}")
#     print(response.text)
    
# t2 = datetime.datetime.now()

# print (t2-t1)