import os
import zipfile
import pandas as pd

DATA_DIR = "/projects/sfcwinds/data/AmeriFlux"  # Where ZIP files are stored
OUTPUT_DIR = "/projects/sfcwinds/data/AmeriFlux"  # Processed files stay here

REQUIRED_COLUMNS = {
    "timestamp": ["TIMESTAMP_START", "TIMESTAMP"],
    "temperature": ["TA_PI_F", "TA", "T_SONIC", "T_SONIC_1_1_1"],
    "wind_speed": ["WS_1_1_1", "WS", "WS_1_2_1"],
    "wind_qc": ["WS_1_1_1_QC", "WS_QC"]
}

skipped_files = []

def process_zip(zip_path):
    """ Process an AmeriFlux ZIP file into a Parquet file. """
    site_id = os.path.basename(zip_path).split("_")[1]  # Extract site ID
    output_file = os.path.join(OUTPUT_DIR, f"{site_id}.parquet")

    if not zipfile.is_zipfile(zip_path):
        print(f"kipping {zip_path}: Not a valid ZIP file")
        skipped_files.append(f"{site_id}: Invalid ZIP file")
        return

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        csv_files = [f for f in zip_ref.namelist() if f.endswith(".csv")]

        if not csv_files:
            print(f"kipping {site_id}: No CSV file found in ZIP")
            skipped_files.append(f"{site_id}: No CSV file in ZIP")
            return

        for csv_file in csv_files:
            with zip_ref.open(csv_file) as f:
                try:
                    df = pd.read_csv(f, comment="#", low_memory=False)

                    available_cols = df.columns.tolist()
                    selected_cols = {}

                    for key, options in REQUIRED_COLUMNS.items():
                        for option in options:
                            if option in available_cols:
                                selected_cols[key] = option
                                break

                    if "temperature" not in selected_cols or "wind_speed" not in selected_cols:
                        print(f"kipping {site_id}: Missing wind or temperature data")
                        skipped_files.append(f"{site_id}: Missing required columns")
                        continue

                    df = df.rename(columns={
                        selected_cols["timestamp"]: "timestamp",
                        selected_cols["temperature"]: "temperature",
                        selected_cols["wind_speed"]: "wind_speed",
                        selected_cols.get("wind_qc", "wind_flag"): "wind_flag"
                    })

                    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y%m%d%H%M", errors="coerce")

                    df = df.dropna(subset=["timestamp"])

                    df.to_parquet(output_file, index=False)
                    print(f"Processed {site_id} -> {output_file}")

                except Exception as e:
                    print(f"rror processing {site_id}: {e}")
                    skipped_files.append(f"{site_id}: {str(e)}")

zip_files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith(".zip")]

print(f"Found {len(zip_files)} ZIP files to process...")

for zip_file in zip_files:
    process_zip(zip_file)

skipped_file_path = os.path.join(OUTPUT_DIR, "skipped_stations.txt")
with open(skipped_file_path, "w") as f:
    for line in skipped_files:
        f.write(line + "\n")

print("AmeriFlux Processing Complete")



