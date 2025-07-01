import pandas as pd
import os

OBS_FILE = "/projects/sfcwinds/data/CO/coagmet_5min.csv"
OUTPUT_DIR = "/projects/sfcwinds/outputs/CO/"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading CoAgMet 5-min data in chunks...")
chunksize = 5000000  

for chunk in pd.read_csv(OBS_FILE, chunksize=chunksize, dtype=str):
    print(f"🔹 Processing {len(chunk)} rows...")

    chunk.columns = ['stationid', 'timestamp', 'airtemperature', 'wind', 'winddirection', 'gust']

    chunk['timestamp'] = pd.to_datetime(chunk['timestamp'], format="%m/%d/%Y %H:%M", errors='coerce')

    numeric_cols = ['airtemperature', 'wind', 'winddirection', 'gust']
    for col in numeric_cols:
        chunk[col] = pd.to_numeric(chunk[col], errors='coerce')  # Convert strings to numbers, set bad values to NaN

    chunk['airtemperature'] = (chunk['airtemperature'] - 32) * 5/9
    chunk['wind'] = chunk['wind'] * 0.44704
    chunk['gust'] = chunk['gust'] * 0.44704

    chunk.replace(-999, pd.NA, inplace=True)

    chunk.dropna(subset=['wind', 'airtemperature'], how='all', inplace=True)

    chunk['year'] = chunk['timestamp'].dt.year

    for (station, year), group in chunk.groupby(['stationid', 'year']):
        station_dir = os.path.join(OUTPUT_DIR, station)
        os.makedirs(station_dir, exist_ok=True)
        parquet_file = os.path.join(station_dir, f"{year}.parquet")

        group.to_parquet(parquet_file, index=False, engine="pyarrow")
        
        print(f"Saved {len(group)} rows to {parquet_file}")

print("Processing complete")



