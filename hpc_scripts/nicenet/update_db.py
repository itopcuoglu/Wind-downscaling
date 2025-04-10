import pandas as pd
import sqlite3


df = pd.read_csv("nice_stations.csv")


df["elev"] = df["elev_ft"] * 0.3048


conn = sqlite3.connect("/kfs2/projects/sfcwinds/data/weather_data.db")
cursor = conn.cursor()


# Loop through each station and update lat, lon, elev
for _, row in df.iterrows():
    cursor.execute("""
        UPDATE stations
        SET lat = ?, lon = ?, elev = ?
        WHERE station_id = ?
    """, (row["lat"], row["lon"], row["elev"], row["station_id"]))


conn.commit()
conn.close()
print("NICEnet station coordinates and elevation updated.")





