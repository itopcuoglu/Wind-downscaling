import os
import sqlite3
import pandas as pd

db_path = "/projects/sfcwinds/data/weather_data.db"
ameriflux_dir = "/projects/sfcwinds/outputs/AmeriFlux/"

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

cursor.execute("PRAGMA table_info(stations)")
station_columns = [col[1] for col in cursor.fetchall()]

required_station_columns = ["station_id", "station_name", "lat", "lon", "height", "elev", "begints", "endts", "source_network", "state"]

missing_columns = [col for col in required_station_columns if col not in station_columns]
if missing_columns:
    raise ValueError(f"Missing columns in 'stations' table: {missing_columns}")

cursor.execute("PRAGMA table_info(data_references)")
data_ref_columns = [col[1] for col in cursor.fetchall()]

for station in os.listdir(ameriflux_dir):
    station_path = os.path.join(ameriflux_dir, station)

    if os.path.isdir(station_path):  # Ensure it's a directory
        metadata_file = os.path.join(station_path, f"{station}.parquet")

        year_files = sorted([f for f in os.listdir(station_path) if f.endswith('.parquet') and f[:4].isdigit()])
        
        if year_files:
            start_year = year_files[0][:4]  # First year
            end_year = year_files[-1][:4]   # Last year
            begints = f"{start_year}-01-01"
            endts = f"{end_year}-12-31"
        else:
            begints, endts = None, None  

        lat, lon, elev = None, None, None
        if os.path.exists(metadata_file):
            df = pd.read_parquet(metadata_file)
            lat = df.get('latitude', [None])[0]
            lon = df.get('longitude', [None])[0]
            elev = df.get('elevation', [None])[0]

        cursor.execute("SELECT * FROM stations WHERE station_id = ?", (station,))
        existing_station = cursor.fetchone()

        if not existing_station:  # Insert missing station
            insert_query = f"INSERT INTO stations ({', '.join(required_station_columns)}) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
            cursor.execute(insert_query, (station, station, lat, lon, 10, elev, begints, endts, "AmeriFlux", "Unknown"))
            print(f"Added station: {station}")

        cursor.execute("SELECT * FROM data_references WHERE station_id = ? AND dataset_source = ?", 
                       (station, "AmeriFlux"))
        existing_data_ref = cursor.fetchone()

        if not existing_data_ref:  # Insert new data reference
            cursor.execute("INSERT INTO data_references VALUES (?, ?, ?, ?, ?)",
                           (station, "AmeriFlux", station_path, begints, endts))
            print(f"Linked data: {station} ({begints} - {endts})")

conn.commit()
conn.close()
print("AmeriFlux data successfully linked to the database")



