# -*- coding: utf-8 -*-
"""
Created on Fri Apr  4 16:25:07 2025

@author: uegerer
"""



import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the JSON data
with open("NM_stations.json", "r") as f:
    data = json.load(f)

# List to store matching station info (STID, wind sensor height, providers)
filtered_station_info = []

# Filter and collect relevant info
for station in data["STATION"]:
    try:
        pos = station["SENSOR_VARIABLES"]["wind_speed"]["wind_speed_set_1"].get("position")
        if pos is not None and float(pos) < 9:
            # Get providers
            providers = station.get("PROVIDERS", [])
            provider_names = [p.get("name", "Unknown") for p in providers]
            provider_str = ", ".join(provider_names) if provider_names else "None"
            
            # Get the PERIOD_OF_RECORD
            period_of_record = station["SENSOR_VARIABLES"]["wind_speed"]["wind_speed_set_1"].get("PERIOD_OF_RECORD", {})
            start_time = period_of_record.get("start", "Unknown")
            end_time = period_of_record.get("end", "Unknown")
            
            # Collect relevant station info
            filtered_station_info.append({
                "STID": station["STID"],
                "name": station["NAME"],
                "state": station["STATE"],
                "network": station["SHORTNAME"],
                "network_long": station["LONGNAME"],
                "Wind Sensor Height (m)": pos,
                "lat": station["LATITUDE"],
                "lon": station["LONGITUDE"],
                #"Provider": provider_str,
                "Start Time": start_time,
                "End Time": end_time
            })
    except (KeyError, TypeError, ValueError):
        continue  # Skip malformed stations

# Print number of stations and the data
print(f"\nTotal stations with wind sensor height < 10 m: {len(filtered_station_info)}")

# Convert list of filtered station info into a pandas DataFrame
df = pd.DataFrame(filtered_station_info)
df["obs_time"] = pd.to_datetime(df["End Time"]) - pd.to_datetime(df["Start Time"])
df['obs_time_years'] = df['obs_time'].dt.total_seconds() / (365.25 * 24 * 3600)


plt.figure()
df.obs_time_years.hist(bins=100, density=False)
plt.ylabel("# of stations")
plt.xlabel("operating time (years)")


# Count occurrences of each provider
provider_counts = df['network'].value_counts()
print(provider_counts)
for provider, count in provider_counts.items():
    print(f"{provider}: {count}")

# Save the dataframe to a CSV file
df.to_csv("NM_stations_under_10m.csv", index=False)

# Save the data as a JSON file
# with open("stations_under_10m.json", "w") as f_out:
#     json.dump(filtered_station_info, f_out, indent=2)