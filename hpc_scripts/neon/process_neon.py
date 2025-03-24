import os
import pandas as pd
import glob

RAW_DIR = "/projects/sfcwinds/data/neon_raw/"
OUTPUT_DIR = "/projects/sfcwinds/outputs/NEON/"

os.makedirs(OUTPUT_DIR, exist_ok=True)

for station in os.listdir(RAW_DIR):
    station_path = os.path.join(RAW_DIR, station)

    if not os.path.isdir(station_path):
        continue  # Skip files, only process directories

    print(f"Processing station: {station}")

    for year in os.listdir(station_path):
        year_path = os.path.join(station_path, year)

        if not os.path.isdir(year_path):
            continue

        all_files = glob.glob(os.path.join(year_path, "*.csv"))

        if not all_files:
            print(f"o CSV files found for {station} in {year}")
            continue

        df_list = []

        for file in all_files:
            try:
                df = pd.read_csv(file)

                df = df.rename(columns={
                    "startDateTime": "timestamp",
                    "windSpeedMean": "windspeed",
                    "windDirMean": "winddirection",
                    "windSpeedMaximum": "gust"
                })

                df = df[["timestamp", "windspeed", "winddirection", "gust"]]

                df["timestamp"] = pd.to_datetime(df["timestamp"])

                df_list.append(df)

            except Exception as e:
                print(f"Error processing {file}: {e}")
        
        if df_list:
            df_year = pd.concat(df_list, ignore_index=True)

            station_output_path = os.path.join(OUTPUT_DIR, station)
            os.makedirs(station_output_path, exist_ok=True)

            output_file = os.path.join(station_output_path, f"{year}.parquet")
            df_year.to_parquet(output_file, index=False)

            print(f"Saved: {output_file}")

print("NEON Processing Complete")



