import sqlite3
import pandas as pd
import os
from config import STATIONS_METADATA_FILE, DB_NAME

def extract_station_metadata():
    """
    Reads station metadata from the CSV file and maps it to the database schema
    """
    if not os.path.exists(STATIONS_METADATA_FILE):
        print(f"Error: Metadata file {STATIONS_METADATA_FILE} not found.")
        return pd.DataFrame()

    try:
        df = pd.read_csv(STATIONS_METADATA_FILE)
    except Exception as e:
        print(f"Error reading {STATIONS_METADATA_FILE}: {e}")
        return pd.DataFrame()

    mapping = {
        "stid": "station_id",
        "source_network": "source_network"
    }
    df.rename(columns=mapping, inplace=True)

    # Ensure that if height or state are missing, they remain null
    required_columns = ["station_id", "station_name", "lat", "lon", "height", "elev", "begints", "endts", "source_network", "state"]
    for col in required_columns:
        if col not in df.columns:
            df[col] = None

    # Reorder columns
    df = df[required_columns].drop_duplicates(subset=["station_id"])
    return df

def insert_station_metadata(stations_df):
    """
    Inserts the station metadata DataFrame into the SQLite database
    """
    if stations_df.empty:
        print("No station metadata found.")
        return

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    for _, row in stations_df.iterrows():
        try:
            cursor.execute('''
                INSERT OR REPLACE INTO stations
                (station_id, station_name, lat, lon, height, elev, begints, endts, source_network, state)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                row["station_id"],
                row["station_name"],
                row["lat"],
                row["lon"],
                row["height"],
                row["elev"],
                row["begints"],
                row["endts"],
                row["source_network"],
                row["state"]
            ))
        except Exception as e:
            print(f"Error inserting station {row['station_id']}: {e}")

    conn.commit()
    conn.close()
    print("Station metadata successfully inserted into the database.")

def main():
    df = extract_station_metadata()
    insert_station_metadata(df)

if __name__ == "__main__":
    main()