import pandas as pd
import sqlite3
import shutil
import os


original_db = "/kfs2/projects/sfcwinds/data/weather_data.db"
expanded_db = "/kfs2/projects/sfcwinds/data/weather_data_expanded.db"
csv_path = "/kfs2/projects/sfcwinds/data/general_scripts/US_stations_under_10m.csv"


if not os.path.exists(expanded_db):
    shutil.copyfile(original_db, expanded_db)
    print("Cloned original database to weather_data_expanded.db")
else:
    print("Expanded DB already exists — skipping clone")


df = pd.read_csv(csv_path)


df = df.rename(columns={
    "STID": "station_id",
    "name": "station_name",
    "lat": "lat",
    "lon": "lon",
    "Wind Sensor Height (m)": "height",
    "Start Time": "begints",
    "End Time": "endts",
    "network": "source_network",
    "state": "state"
})


df["elev"] = None  


conn = sqlite3.connect(expanded_db)
cursor = conn.cursor()


added_count = 0
for _, row in df.iterrows():
    try:
        cursor.execute("""
            INSERT OR IGNORE INTO stations 
            (station_id, station_name, lat, lon, height, elev, begints, endts, source_network, state)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
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
        added_count += 1
    except Exception as e:
        print(f"Failed to insert {row['station_id']}: {e}")


conn.commit()
conn.close()
print(f"Finished. Added {added_count} Synoptic stations to weather_data_expanded.db")

