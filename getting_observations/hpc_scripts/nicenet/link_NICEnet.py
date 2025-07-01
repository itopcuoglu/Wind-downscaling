import os
import sqlite3
import pandas as pd


DB_PATH = "/kfs2/projects/sfcwinds/data/weather_data.db"
STATIONS_CSV = "/kfs2/projects/sfcwinds/data/stations.csv"
NICENET_DIR = "/kfs2/projects/sfcwinds/outputs/NICEnet"


conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()


stations_csv_df = pd.read_csv(STATIONS_CSV)
stations_csv_df["stid"] = stations_csv_df["stid"].str.upper()


for station_id in os.listdir(NICENET_DIR):
    station_id = station_id.upper()


    cursor.execute("SELECT 1 FROM stations WHERE station_id = ?", (station_id,))
    if cursor.fetchone():
        print(f"{station_id} already in database.")
        continue


    row = stations_csv_df[stations_csv_df["stid"] == station_id]
    if not row.empty:
        row = row.iloc[0]
        station_name = row["station_name"]
        lat = row["lat"]
        lon = row["lon"]
        height = row["height"]
        elev = row["elev"]
        begints = row["begints"]
        endts = row["endts"]
        state = row["state"]
    else:
        station_name = None
        lat = None
        lon = None
        height = None
        elev = None
        begints = None
        endts = None
        state = None


    cursor.execute("""
        INSERT INTO stations (
            station_id, station_name, lat, lon, height, elev, begints, endts, source_network, state
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (station_id, station_name, lat, lon, height, elev, begints, endts, "NICEnet", state))


    print(f"Inserted {station_id} into database.")


conn.commit()
conn.close()





