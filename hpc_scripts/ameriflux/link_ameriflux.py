import os
import sqlite3
import pandas as pd


DB_PATH = "/kfs2/projects/sfcwinds/data/weather_data.db"
AMERIFLUX_DIR = "/kfs2/projects/sfcwinds/outputs/AmeriFlux"
NETWORK = "AmeriFlux"


conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()


for station_id in os.listdir(AMERIFLUX_DIR):
    station_path = os.path.join(AMERIFLUX_DIR, station_id)
    if not os.path.isdir(station_path):
        continue


    cursor.execute("SELECT 1 FROM stations WHERE station_id = ?", (station_id,))
    if cursor.fetchone():
        print(f"Station {station_id} already in database.")
        continue


    parquet_files = [f for f in os.listdir(station_path) if f.endswith(".parquet")]
    if not parquet_files:
        print(f"o data files for {station_id}. Skipping.")
        continue


    try:
        df = pd.read_parquet(os.path.join(station_path, parquet_files[0]))
        lat = df.get("latitude", pd.Series([None])).iloc[0]
        lon = df.get("longitude", pd.Series([None])).iloc[0]
    except Exception as e:
        print(f"ailed to read {station_id}: {e}")
        continue


    state = station_id.split("-")[1][:2] if "-" in station_id else "NA"


    cursor.execute("""
        INSERT INTO stations (station_id, station_name, lat, lon, height, elev, begints, endts, source_network, state)
        VALUES (?, ?, ?, ?, NULL, NULL, NULL, NULL, ?, ?)
    """, (station_id, station_id, lat, lon, NETWORK, state))


    print(f"Added {station_id} to database.")


conn.commit()
conn.close()





