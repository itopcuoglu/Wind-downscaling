import sqlite3
import requests
import pandas as pd


TOKEN = ""


response = requests.get("https://api.synopticdata.com/v2/networks", params={"token": TOKEN})
synoptic_data = response.json()


synoptic_networks = {n["NAME"].lower() for n in synoptic_data.get("NETWORKS", [])}


conn = sqlite3.connect("/kfs2/projects/sfcwinds/data/weather_data.db")
df = pd.read_sql_query("SELECT DISTINCT source_network FROM stations", conn)
conn.close()


db_networks = {x.lower() for x in df['source_network'].dropna()}


only_in_db = sorted(db_networks - synoptic_networks)
only_in_synoptic = sorted(synoptic_networks - db_networks)


print("\nNetworks in our database NOT in Synoptic")
for name in only_in_db:
    print(name)


print("\nNetworks in Synoptic NOT in our database")
for name in only_in_synoptic:
    print(name)



if "NETWORKS" not in synoptic_data:
    print("Warning: No networks returned from Synoptic.")
    print("API Response:", synoptic_data)
else:
    synoptic_networks = {n["NAME"].lower() for n in synoptic_data["NETWORKS"]}








