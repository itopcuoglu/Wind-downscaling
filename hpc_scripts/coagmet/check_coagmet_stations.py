import pandas as pd
import sqlite3

OBS_FILE = "/projects/sfcwinds/data/CO/coagmet_5min_2010_2025_cleaned.csv"
DB_PATH = "/projects/sfcwinds/data/weather_data.db"

print("Loading cleaned CoAgMet observation data...")
df_obs = pd.read_csv(OBS_FILE, skiprows=2, low_memory=False)

# Standardize column names
df_obs.columns = df_obs.columns.str.strip().str.lower()
station_col = [col for col in df_obs.columns if "station" in col][0]
df_obs[station_col] = df_obs[station_col].str.upper().str.strip()

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

print("\n Querying station metadata from database...")
df_stations = pd.read_sql("SELECT station_id FROM stations", conn)
df_stations['station_id'] = df_stations['station_id'].str.upper().str.strip()

conn.close()

# Compare station IDs
obs_stations = set(df_obs[station_col].unique())
db_stations = set(df_stations['station_id'].unique())

matched_stations = obs_stations.intersection(db_stations)
missing_stations = obs_stations - db_stations

print(f"\n Total CoAgMet Stations in 5-min Dataset: {len(obs_stations)}")
print(f" Stations Found in Database: {len(matched_stations)}")
print(f"⚠Missing Stations (Not in Database): {len(missing_stations)}")

if missing_stations:
    print("\n Sample of Missing Stations:")
    print(list(missing_stations)[:10])  # Show first 10 for readability

MISSING_FILE = "/projects/sfcwinds/data/missing_coagmet_stations.csv"
pd.DataFrame(missing_stations, columns=['station_id']).to_csv(MISSING_FILE, index=False)
print(f"\n Missing station IDs saved to: {MISSING_FILE}")




