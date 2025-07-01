import os
import pandas as pd
import random


outputs_dir = "/kfs2/projects/sfcwinds/outputs"
networks_with_qc = []
networks_without_qc = []


for network in os.listdir(outputs_dir):
    network_path = os.path.join(outputs_dir, network)
    if not os.path.isdir(network_path):
        continue


    stations = [s for s in os.listdir(network_path) if os.path.isdir(os.path.join(network_path, s))]
    sample_stations = random.sample(stations, min(3, len(stations)))
    has_qc = False


    for station in sample_stations:
        station_path = os.path.join(network_path, station)
        parquet_files = [f for f in os.listdir(station_path) if f.endswith(".parquet")]
        if not parquet_files:
            continue
        first_file = parquet_files[0]
        file_path = os.path.join(station_path, first_file)
        try:
            df = pd.read_parquet(file_path, columns=None)
            if 'qualitycontrol' in df.columns:
                has_qc = True
                break
        except Exception as e:
            print(f"Error reading {file_path}: {e}")


    if has_qc:
        networks_with_qc.append(network)
    else:
        networks_without_qc.append(network)


print("Networks WITH 'qualitycontrol' column:")
for net in networks_with_qc:
    print("  •", net)


print("\nNetworks WITHOUT 'qualitycontrol' column:")
for net in networks_without_qc:
    print("  •", net)





