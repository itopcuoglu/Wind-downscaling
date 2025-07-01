import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from datetime import datetime


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# file path below can point to either initial database or expanded
# file_path = os.path.join(BASE_DIR, "..", "data", "weather_data.db")


df = pd.read_csv(file_path)


df_clean = df.copy()
df_clean['elev'] = pd.to_numeric(df_clean['elev'], errors='coerce')


# Parse years
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


df_clean = df_clean[df_clean['state'].isin(conus_states)]
df_clean = df_clean[df_clean['lat'] < 49.5]  


total_stations = len(df_clean)
num_states = df_clean['state'].nunique()
avg_duration = df_clean['duration_years'].mean()
avg_elev = df_clean['elev'].mean()
avg_height = df_clean['height'].mean()


state_counts = df_clean['state'].value_counts().reset_index()
state_counts.columns = ['state', 'count']
bar_fig = px.bar(state_counts, x='state', y='count',
                 title="Stations per State",
                 labels={'count': 'Number of Stations'})


source_counts = df_clean['source_network'].value_counts().reset_index()
source_counts.columns = ['source_network', 'count']
pie_fig = px.pie(source_counts, names='source_network', values='count',
                 title="Stations by Source Network")
pie_fig.update_traces(textinfo='none')


duration_hist = px.histogram(df_clean, x='duration_years', nbins=30,
                             title="Distribution of Operating Durations",
                             labels={'duration_years': 'Years', 'count': 'Number of Stations'})


summary_table = go.Table(
    header=dict(values=["Metric", "Value"],
                fill_color='paleturquoise',
                align='left'),
    cells=dict(values=[
        ["Total Stations", "Number of States", "Average Operating Duration",
         "Average Elevation (m)", "Average Height (m)"],
        [total_stations, num_states, f"{avg_duration:.1f} years",
         f"{avg_elev:.1f}", f"{avg_height:.1f}"]
    ],
    fill_color='lavender',
    align='left')
)


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


map_fig.update_traces(marker=dict(size=6, opacity=0.8,
                                  line=dict(width=0.3, color='black')))
map_fig.update_layout(geo=dict(showland=True, landcolor='lightgray',
                               showlakes=True, lakecolor='lightblue'))


fig = make_subplots(
    rows=3, cols=2,
    subplot_titles=("US Map", "Stations per State",
                    "Source Network Distribution", "Operating Duration Distribution",
                    "Summary Metrics"),
    specs=[
        [{"type": "geo"}, {"type": "xy"}],
        [{"type": "domain"}, {"type": "xy"}],
        [{"colspan": 2, "type": "table"}, None]
    ],
    vertical_spacing=0.1
)


for trace in map_fig.data:
    fig.add_trace(trace, row=1, col=1)
fig.update_geos(scope='usa', row=1, col=1)


for trace in bar_fig.data:
    fig.add_trace(trace, row=1, col=2)


for trace in pie_fig.data:
    fig.add_trace(trace, row=2, col=1)


for trace in duration_hist.data:
    fig.add_trace(trace, row=2, col=2)


fig.add_trace(summary_table, row=3, col=1)


fig.update_layout(
    height=1000,
    title_text="Wind Observation Stations Dashboard",
    showlegend=False
)


output_html = os.path.join(BASE_DIR, "dashboard_overview.html")
fig.write_html(output_html, auto_open=True)




