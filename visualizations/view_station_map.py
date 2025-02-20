import pandas as pd
import plotly.express as px
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(BASE_DIR, "..", "data", "stations.csv")

df = pd.read_csv(file_path)

if not {"lat", "lon", "station_name", "state", "elev", "stid", "height"}.issubset(df.columns):
    raise ValueError("Missing required columns: lat, lon, station_name, state, elev, stid, height")

df["hover_info"] = (
    df["station_name"] + " (" + df["stid"] + ")<br>" +
    "State: " + df["state"] + "<br>" +  
    "Elevation: " + df["elev"].astype(str) + "m<br>" +
    "Latitude: " + df["lat"].astype(str) + "<br>" +
    "Longitude: " + df["lon"].astype(str) + "<br>" +
    "Height: " + df["height"].astype(str) + "m"
)

fig = px.scatter_geo(
    df,
    lat="lat",
    lon="lon",
    hover_name="station_name",
    hover_data={"lat": False, "lon": False, "stid": False, "hover_info": True}, 
    projection="albers usa",
    title="Weather Station Locations"
)

fig.update_traces(marker=dict(
    size=6,  
    color='cornflowerBlue',  
    opacity=0.8,  
    line=dict(width=0.3, color='black')  
))

fig.update_traces(
    hoverlabel=dict(
        bgcolor="PowderBlue",  
        font_size=12,  
        font_family="Arial",  
        font_color="black"  
    )
)

fig.update_layout(
    geo=dict(
        showland=True,
        landcolor='lightgray',
        showlakes=True,
        lakecolor='lightblue'
    ),
    margin={"r":0,"t":30,"l":0,"b":0}  
)

fig.show()



