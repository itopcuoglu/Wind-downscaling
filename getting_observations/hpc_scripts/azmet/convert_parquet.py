import pandas as pd
import os
from datetime import datetime, timedelta

INPUT_DIR = "/projects/sfcwinds/outputs/AZMet/"
OUTPUT_DIR = "/projects/sfcwinds/outputs/AZMet/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

STATION_NAME_MAP = {
    "01": "Aguila",
    "02": "Bonita",
    "04": "Safford",
    "05": "Coolidge",
    "06": "Maricopa",
    "07": "Desert Ridge",
    "08": "Parker",
    "09": "Bonita",
    "12": "Phoenix Greenway",
    "14": "Yuma North Gila",
    "15": "Phoenix Encanto",
    "19": "Paloma",
    "20": "Mohave",
    "22": "Queen Creek",
    "23": "Harquahala",
    "24": "Roll",
    "26": "Buckeye",
    "27": "Desert Ridge",
    "28": "Mohave #2",
    "32": "Payson",
    "33": "Bowie",
    "35": "Parker #2",
    "36": "Yuma South",
    "37": "San Simon",
    "38": "Sahuarita",
    "39": "Willcox Bench",
    "40": "Ft Mohave CA",
    "41": "Salome"
}

COLUMNS = [
    "timestamp", "windspeed", "winddirection", "gust", "temperature"
]

def convert_to_timestamp(year, doy, hour):
    try:
        year, doy, hour = int(year), int(doy), int(hour)  
        base_date = datetime(year, 1, 1) + timedelta(days=doy - 1, hours=hour)
        return base_date.strftime("%Y-%m-%d %H:%M:%S")  # Convert to string timestamp
    except ValueError:
        return None  # Handle invalid date cases

def process_file(file_path):
    try:
        filename = os.path.basename(file_path)
        if not filename.endswith("rh.txt"):  #
            print(f"Skipping non-hourly file: {filename}")
            return
        
        station_id = filename[:2]  
        year_part = filename[2:4]  #
        full_year = 1900 + int(year_part) if int(year_part) > 50 else 2000 + int(year_part)  
        station_name = STATION_NAME_MAP.get(station_id, f"Unknown_{station_id}")
        
        print(f"Processing {file_path} for station {station_name} ({station_id}) year {full_year}")
        
        df = pd.read_csv(file_path, header=None, on_bad_lines='skip')
        
        df = df.iloc[:, [0, 1, 2, 10, 12, 14, 3]]  
        df.columns = ["year", "doy", "hour", "windspeed", "winddirection", "gust", "temperature"]
        
        df["year"] = df["year"].astype(int)
        df["doy"] = df["doy"].astype(int)
        df["hour"] = df["hour"].astype(int)
        
        df["timestamp"] = df.apply(lambda row: convert_to_timestamp(row["year"], row["doy"], row["hour"]), axis=1)
        
        df = df[["timestamp", "windspeed", "winddirection", "gust", "temperature"]]  
        station_output_dir = os.path.join(OUTPUT_DIR, station_name)
        os.makedirs(station_output_dir, exist_ok=True)
        
        output_file = os.path.join(station_output_dir, f"{full_year}.parquet")
        df.to_parquet(output_file, engine="pyarrow", index=False)
        print(f"Processed and saved: {output_file}")
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

for file in os.listdir(INPUT_DIR):
    if file.endswith(".txt"):
        process_file(os.path.join(INPUT_DIR, file))

print("AZMet data reformatted with correct column order!")


