import os
import pandas as pd
from config import OUTPUT_PARQUET_DIR

files = [f for f in os.listdir(OUTPUT_PARQUET_DIR) if f.endswith('.parquet')]
if files:
    for file in files:
        file_path = os.path.join(OUTPUT_PARQUET_DIR, file)
        print(f"\nPreview of {file}: ")
        df = pd.read_parquet(file_path)
        print(df.head())
else:
    print("No Parquet files found in", OUTPUT_PARQUET_DIR)
    