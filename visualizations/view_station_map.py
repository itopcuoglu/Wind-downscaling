import pandas as pd
import plotly.express as px
import os
from datetime import datetime


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# file path below can point to either initial database or expanded
# file_path = os.path.join(BASE_DIR, "..", "data", "weather_data.db")


df = pd.read_csv(file_path)


df_clean = df[df['elev'] != 'elev'].copy()
df_clean['elev'] = pd.to_numeric(df_clean['elev'], errors='coerce')


current_year = datetime.now().year


def parse_year(x):
    s = str(x).strip()
    return int(s[:4]) if s and s[:4].isdigit() else current_year


df_clean['beg_year'] = pd.to_numeric(df_clean['begints'].astype(str).str[:4], errors='coerce')
df_clean['end_year'] = df_clean['endts'].fillna('').astype(str).apply(parse_year)
df_clean = df_clean.dropna(subset=['beg_year'])


df_clean['duration_years'] = df_clean['end_year'] - df_clean['beg_year']
df_clean['height'] = pd.to_numeric(df_clean['height'], errors='coerce')


conus_states = ['AL', 'AR', 'AZ', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'IA',
                'ID', 'IL', 'IN', 'KS', 'KY', 'LA', 'MA', 'MD', 'ME', 'MI',
                'MN', 'MO', 'MS', 'MT', 'NC', 'ND', 'NE', 'NH', 'NJ', 'NM',
                'NV', 'NY', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN',
                'TX', 'UT', 'VA', 'VT', 'WA', 'WI', 'WV', 'WY']


df_clean = df_clean[df_clean['state'].isin(conus_states)].copy()


df_clean = df_clean[df_clean['lat'] < 49.5]


required_columns = {"lat", "lon", "station_name", "state", "elev", "station_id", "height"}
if not required_columns.issubset(df_clean.columns):
    missing = required_columns - set(df_clean.columns)
    raise ValueError("Missing required columns: " + ", ".join(missing))


df_clean["hover_info"] = (
    df_clean["station_name"] + " (" + df_clean["station_id"] + ")<br>" +
    "State: " + df_clean["state"] + "<br>" +
    "Elevation: " + df_clean["elev"].astype(str) + "m<br>" +
    "Latitude: " + df_clean["lat"].astype(str) + "<br>" +
    "Longitude: " + df_clean["lon"].astype(str) + "<br>" +
    "Height: " + df_clean["height"].astype(str) + "m"
)


map_fig = px.scatter_geo(
    df_clean,
    lat="lat",
    lon="lon",
    color="state",
    hover_name="station_name",
    hover_data={"lat": False, "lon": False, "station_id": False, "hover_info": True},
    projection="albers usa",
    title="Station Locations on US Map"
)


map_fig.update_traces(marker=dict(
    size=6,
    opacity=0.8,
    line=dict(width=0.3, color='black')
))


map_fig.update_layout(
    geo=dict(
        showland=True,
        landcolor='lightgray',
        showlakes=True,
        lakecolor='lightblue'
    ),
    margin={"r": 0, "t": 30, "l": 0, "b": 0}
)


output_html = os.path.join(BASE_DIR, "station_map.html")
map_fig.write_html(output_html, auto_open=True)




