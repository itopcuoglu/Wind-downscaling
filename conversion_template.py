#!/usr/bin/env python3
"""
This script reads in a CSV file containing station metadata (which may have
various column names and formats), renames the columns to a standardized set, adds
a default state name if desired, converts date columns to a specified format, and
saves the output as a new CSV file.

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

Usage Example:
    python standardize_metadata.py input_metadata.csv standardized_metadata.csv \
        --default_state "CA" --date_format "%Y-%m-%d"
"""

import pandas as pd
import argparse

# Define a mapping from input column names to our standardized column names.
# **Adjust these keys/values based on the source file's header**
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

def standardize_metadata(input_file, output_file, mapping, default_state, date_format):
    """
    Reads a CSV file, renames columns according to the mapping provided, fills in the
    state column if a default is provided, converts date fields, and writes out a new CSV.
    """
    try:
        df = pd.read_csv(input_file)
    except Exception as e:
        print(f"Error reading {input_file}: {e}")
        return

    # Rename columns based on mapping
    rename_dict = {col: mapping[col] for col in df.columns if col in mapping}
    df.rename(columns=rename_dict, inplace=True)

    # Fill or create the state column if default_state is provided.
    if default_state:
        if 'state' in df.columns:
            # Fill missing or empty state values with the default state.
            df['state'] = df['state'].fillna(default_state)
            df.loc[df['state'].astype(str).str.strip() == '', 'state'] = default_state
        else:
            df['state'] = default_state

    # Convert date columns if present.
    for date_col in ['begints', 'endts']:
        if date_col in df.columns:
            try:
                # Parse dates and reformat them
                df[date_col] = pd.to_datetime(df[date_col], errors='coerce').dt.strftime(date_format)
            except Exception as e:
                print(f"Error converting {date_col}: {e}")

    try:
        df.to_csv(output_file, index=False)
        print(f"Standardized metadata saved to {output_file}")
    except Exception as e:
        print(f"Error saving standardized metadata to {output_file}: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Standardize station metadata CSV files.")
    parser.add_argument("input_file", help="Path to the input CSV file with station metadata")
    parser.add_argument("output_file", help="Path to save the standardized output CSV file")
    parser.add_argument("--default_state", help="Default state name to add to the state column (if missing)", default=None)
    parser.add_argument("--date_format", help="Desired date format (e.g., '%Y-%m-%d')", default="%Y-%m-%d")
    args = parser.parse_args()

    standardize_metadata(args.input_file, args.output_file, COLUMN_MAPPING, args.default_state, args.date_format)



