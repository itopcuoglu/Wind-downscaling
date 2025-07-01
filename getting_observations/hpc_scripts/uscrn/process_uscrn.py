import os
import pandas as pd
import glob

DATA_DIR = "/projects/sfcwinds/data/USCRN/"
HEADERS_FILE = os.path.join(DATA_DIR, "HEADERS.txt")
OUTPUT_DIR = "/projects/sfcwinds/outputs/USCRN/"

os.makedirs(OUTPUT_DIR, exist_ok=True)

KEEP_COLS = ["UTC_DATE", "UTC_TIME", "AIR_TEMPERATURE", "WIND_1_5", "WIND_FLAG"]

with open(HEADERS_FILE, "r") as f:
    headers = f.readline().strip().split() 
    f.readline()  

print("Headers Found:", headers)

try:
    col_indices = [headers.index(col.strip()) for col in KEEP_COLS]
except ValueError as e:
    print(f"Error: One of the columns is missing in headers.txt: {e}")
    exit(1)

for year_folder in sorted(os.listdir(DATA_DIR)):
    year_path = os.path.join(DATA_DIR, year_folder)

    if not os.path.isdir(year_path) or not year_folder.isdigit():
        continue
    if int(year_folder) < 2014:
        continue

    print(f"Processing year: {year_folder}")

    year_output_dir = os.path.join(OUTPUT_DIR, year_folder)
    os.makedirs(year_output_dir, exist_ok=True)

    for file_path in glob.glob(os.path.join(year_path, "*.txt")):
        try:
            df = pd.read_csv(
                file_path,
                delim_whitespace=True,
                header=None,
                usecols=col_indices,
                names=KEEP_COLS,
                dtype=str  
            )

            df["timestamp"] = pd.to_datetime(df["UTC_DATE"] + df["UTC_TIME"], format="%Y%m%d%H%M")

            df["airtemperature"] = pd.to_numeric(df["AIR_TEMPERATURE"], errors="coerce")
            df["wind"] = pd.to_numeric(df["WIND_1_5"], errors="coerce")
            df["qualitycontrol"] = pd.to_numeric(df["WIND_FLAG"], errors="coerce")

            df.drop(columns=["UTC_DATE", "UTC_TIME", "AIR_TEMPERATURE", "WIND_1_5", "WIND_FLAG"], inplace=True)

            station_id = os.path.basename(file_path).split("-")[-1].replace(".txt", "")

            df.insert(0, "stationid", station_id)

            output_file = os.path.join(year_output_dir, f"{station_id}.parquet")
            df.to_parquet(output_file, index=False, engine="pyarrow")

            print(f"Saved: {output_file}")

        except Exception as e:
            print(f"rror processing {file_path}: {e}")

print("Processing complete")



