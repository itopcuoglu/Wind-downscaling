import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from datetime import datetime

# -------------------------
# Load and Clean the Data
# -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(BASE_DIR, "..", "data", "stations.csv")

# Read the CSV file
df = pd.read_csv(file_path)

# Remove rows with an extra header (rows where the 'elev' column equals the literal string "elev")
df_clean = df[df['elev'] != 'elev'].copy()

# Convert the 'elev' column to numeric using .loc to avoid SettingWithCopyWarning
df_clean.loc[:, 'elev'] = pd.to_numeric(df_clean['elev'], errors='coerce')

# -------------------------
# Process Date Fields & Operating Duration
# -------------------------
current_year = datetime.now().year

def parse_year(x):
    s = str(x).strip()
    return int(s[:4]) if s and s[:4].isdigit() else current_year

# Extract the beginning year from the 'begints' column
df_clean['beg_year'] = pd.to_numeric(df_clean['begints'].astype(str).str[:4], errors='coerce')

# For 'endts', if missing or invalid, use the current year
df_clean['end_year'] = df_clean['endts'].fillna('').astype(str).apply(parse_year)

# Drop rows with invalid beginning years
df_clean = df_clean.dropna(subset=['beg_year'])

# Calculate operating duration (in years)
df_clean['duration_years'] = df_clean['end_year'] - df_clean['beg_year']

# -------------------------
# Summary Metrics
# -------------------------
total_stations = len(df_clean)
num_states = df_clean['state'].nunique()
avg_duration = df_clean['duration_years'].mean()
avg_elev = df_clean['elev'].mean()

# -------------------------
# Create Individual Charts
# -------------------------
# Bar Chart: Stations per State
state_counts = df_clean['state'].value_counts().reset_index()
state_counts.columns = ['state', 'count']
bar_fig = px.bar(state_counts, x='state', y='count',
                 title="Stations per State",
                 labels={'count': 'Number of Stations'})

# Pie Chart: Distribution by Source Network
source_counts = df_clean['source_network'].value_counts().reset_index()
source_counts.columns = ['source_network', 'count']
pie_fig = px.pie(source_counts, names='source_network', values='count',
                 title="Stations by Source Network")

# Map: Station Locations on US Map (colored by elevation)
map_fig = px.scatter_geo(df_clean,
                         lat='lat',
                         lon='lon',
                         scope='usa',
                         color='elev',
                         title="Station Locations on US Map",
                         hover_name='station_name',
                         hover_data=['state', 'duration_years', 'elev'])

# Histogram: Distribution of Operating Durations
duration_hist = px.histogram(df_clean, x='duration_years', nbins=30,
                             title="Distribution of Operating Durations (Years)",
                             labels={'duration_years': 'Operating Duration (Years)'})

# Summary Table: Key Statistics
summary_table = go.Table(
    header=dict(values=["Metric", "Value"],
                fill_color='paleturquoise',
                align='left'),
    cells=dict(values=[
        ["Total Stations", "Number of States", "Average Operating Duration", "Average Elevation"],
        [total_stations, num_states, f"{avg_duration:.1f} years", f"{avg_elev:.1f}"]
    ],
    fill_color='lavender',
    align='left')
)

# -------------------------
# Build the Dashboard Layout
# -------------------------
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

# Add US Map (row 1, col 1)
for trace in map_fig.data:
    fig.add_trace(trace, row=1, col=1)
fig.update_geos(scope='usa', row=1, col=1)

# Add Bar Chart (row 1, col 2)
for trace in bar_fig.data:
    fig.add_trace(trace, row=1, col=2)

# Add Pie Chart (row 2, col 1)
for trace in pie_fig.data:
    fig.add_trace(trace, row=2, col=1)

# Add Histogram (row 2, col 2)
for trace in duration_hist.data:
    fig.add_trace(trace, row=2, col=2)

# Add Summary Table (row 3, col 1 spanning 2 columns)
fig.add_trace(summary_table, row=3, col=1)

fig.update_layout(
    height=1000,
    title_text="Wind Observation Stations Dashboard",
    showlegend=False
)

fig.show()



