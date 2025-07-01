import pandas as pd
import os
from config import DATA_FILE, STATIONS_METADATA_FILE, OUTPUT_PARQUET_DIR

# Define the column mapping for observation data
OBS_COLUMN_MAPPING = {
    "station": "station_id",   
    "valid": "time",           
    "tmpf": "temperature",     
    "drct": "wind_direction",  
    "sped": "wind_speed",      
    "gust_mph": "wind_gusts"
}

# Conversion functions
def fahrenheit_to_celsius(f):
    try:
        return (float(f) - 32) * 5.0 / 9.0
    except:
        return None

def mph_to_ms(mph):
    try:
        return float(mph) * 0.44704
    except:
        return None

def process_observations():
    try:
        obs_df = pd.read_csv(DATA_FILE)
    except Exception as e:
        print(f"Error reading observation data file {DATA_FILE}: {e}")
        return None

    # Rename columns as per our mapping
    obs_df.rename(columns=OBS_COLUMN_MAPPING, inplace=True)

    # Convert units
    if "temperature" in obs_df.columns:
        obs_df["temperature"] = obs_df["temperature"].apply(fahrenheit_to_celsius)
    if "wind_speed" in obs_df.columns:
        obs_df["wind_speed"] = obs_df["wind_speed"].apply(mph_to_ms)
    if "wind_gusts" in obs_df.columns:
        obs_df["wind_gusts"] = pd.to_numeric(obs_df["wind_gusts"], errors="coerce").apply(mph_to_ms)

    # Ensure we have the *required* observation columns
    required_obs_columns = ["station_id", "time", "wind_speed", "wind_direction", "wind_gusts", "temperature"]
    for col in required_obs_columns:
        if col not in obs_df.columns:
            obs_df[col] = None

    return obs_df

def add_state_to_observations(obs_df):
    """
    Joins the observation DataFrame with the station metadata to add the state for each observation
    """
    try:
        meta_df = pd.read_csv(STATIONS_METADATA_FILE)
    except Exception as e:
        print(f"Error reading station metadata from {STATIONS_METADATA_FILE}: {e}")
        return obs_df

    # Rename metadata columns to match our join keys if needed
    mapping = {
        "stid": "station_id",
        "state": "state"
    }
    meta_df.rename(columns=mapping, inplace=True)
    meta_subset = meta_df[["station_id", "state"]].drop_duplicates()

    obs_df = obs_df.merge(meta_subset, on="station_id", how="left")
    return obs_df

def save_observations_by_state(obs_df):
    """
    Splits the observation DataFrame by state and writes each group to a separate Parquet file
    If state is missing, we can put it in a file like 'observations_unknown.parquet'
    """
    if not os.path.exists(OUTPUT_PARQUET_DIR):
        os.makedirs(OUTPUT_PARQUET_DIR)

    for state, group in obs_df.groupby("state"):
        state_label = state if pd.notnull(state) else "unknown"
        output_file = os.path.join(OUTPUT_PARQUET_DIR, f"observations_{state_label}.parquet")
        try:
            group.to_parquet(output_file, index=False)
            print(f"Saved {len(group)} observations for state {state_label} to {output_file}.")
        except Exception as e:
            print(f"Error saving observations for state {state_label}: {e}")

def main():
    obs_df = process_observations()
    if obs_df is None:
        return
    # Add state information by joining with station metadata
    obs_df = add_state_to_observations(obs_df)
    # Save the observations split by state
    save_observations_by_state(obs_df)

if __name__ == "__main__":
    main()