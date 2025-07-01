import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


base_dir = "/kfs2/projects/sfcwinds/outputs"
networks = ["AmeriFlux", "AZMet", "CoAgMet", "NEON", "NICEnet", "USCRN"]


for network in networks:
    network_path = os.path.join(base_dir, network)
    if not os.path.exists(network_path):
        continue


    for station in tqdm(os.listdir(network_path), desc=f"Scanning {network}"):
        station_path = os.path.join(network_path, station)
        if not os.path.isdir(station_path):
            continue


        for file in os.listdir(station_path):
            if file.endswith(".parquet"):
                file_path = os.path.join(station_path, file)


                try:
                    df = pd.read_parquet(file_path)


                    if pd.api.types.is_datetime64tz_dtype(df["timestamp"]):
                        df["timestamp"] = df["timestamp"].dt.tz_localize(None)


                        df.to_parquet(file_path, index=False)
                        print(f"ixed timestamp in: {file_path}")
                except Exception as e:
                    print(f"Failed to read/fix {file_path}: {e}")





