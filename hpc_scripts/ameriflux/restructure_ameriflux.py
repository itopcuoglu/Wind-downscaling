import pandas as pd
import os
import glob

input_dir = "/projects/sfcwinds/data/AmeriFlux_cleaned"
output_dir = "/projects/sfcwinds/outputs/AmeriFlux"

os.makedirs(output_dir, exist_ok=True)

parquet_files = glob.glob(f"{input_dir}/*.parquet")

for file in parquet_files:
    station_id = os.path.basename(file).replace(".parquet", "")
    station_folder = os.path.join(output_dir, station_id)

    os.makedirs(station_folder, exist_ok=True)

    df = pd.read_parquet(file)

    df["year"] = df["timestamp"].dt.year

    # Split by year and save
    for year, year_df in df.groupby("year"):
        output_file = os.path.join(station_folder, f"{year}.parquet")
        year_df.drop(columns=["year"]).to_parquet(output_file, index=False)
        print(f"Saved {output_file}")

print("Restructuring Complete")



