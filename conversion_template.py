#!/usr/bin/env python3
"""
This script reads in a CSV file containing station metadata (which may have
various column names and formats), renames the columns to a standardized set, and
saves the output as a new CSV file.

Many CSV's will include different names for their columns (i.e, "tmp" for "temperature" or "gst" for "gust").
The column mappings below are meant to be manually changed in order to convert them to our format.

Standardized columns include:
    - station_id
    - station_name
    - lat
    - lon
    - height
    - elev
    - begints
    - endts
    - source_network
    - state

You can adjust the mapping below to suit each input file's specific column names.
"""

import pandas as pd
import argparse

# Define a mapping from input column names to our standardized column names.
# Adjust these keys/values based on the source file's header
# For example, if one file uses "id" for station ID, we map it to "station_id"
COLUMN_MAPPING = {
    'id': 'station_id',
    'name': 'station_name',
    'latitude': 'lat',
    'longitude': 'lon',
    'measurement_height': 'height',
    'elevation': 'elev',
    'start_time': 'begints',
    'end_time': 'endts',
    'network': 'source_network',
    'state': 'state'
}

def standardize_metadata(input_file, output_file, mapping=COLUMN_MAPPING):
    """
    Reads a CSV file, renames columns according to the mapping provided, and
    writes out a new CSV file with the standardized header.
    """
    try:
        df = pd.read_csv(input_file)
    except Exception as e:
        print(f"Error reading {input_file}: {e}")
        return

    # Create a dictionary for renaming only the columns that match our mapping keys.
    rename_dict = {col: mapping[col] for col in df.columns if col in mapping}
    df.rename(columns=rename_dict, inplace=True)

    try:
        df.to_csv(output_file, index=False)
        print(f"Standardized metadata saved to {output_file}")
    except Exception as e:
        print(f"Error saving standardized metadata to {output_file}: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Standardize station metadata CSV files.")
    parser.add_argument("input_file", help="Path to the input CSV file with station metadata")
    parser.add_argument("output_file", help="Path to save the standardized output CSV file")
    args = parser.parse_args()

    standardize_metadata(args.input_file, args.output_file)



