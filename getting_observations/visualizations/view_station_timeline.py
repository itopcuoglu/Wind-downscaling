import pandas as pd
import plotly.graph_objects as go
import os
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(BASE_DIR, "..", "data", "stations.csv")
df = pd.read_csv(file_path)

current_year = datetime.now().year

def parse_year(x):
    s = str(x).strip()
    if s == '' or not s[:4].isdigit():
        return current_year
    return int(s[:4])

df['beg_year'] = pd.to_numeric(df['begints'].astype(str).str[:4], errors='coerce')
df['end_year'] = df['endts'].fillna('').astype(str).apply(parse_year)

df = df.dropna(subset=['beg_year'])

df['duration_years'] = df['end_year'] - df['beg_year']

max_duration = df['duration_years'].max()
bins = [0, 1, 2, 5, 10, 20, max_duration + 1]
labels = ['1 year', '2 years', '5 years', '10 years', '20 years', '20+ years']
df['duration_bin'] = pd.cut(df['duration_years'], bins=bins, labels=labels, right=False)

states = sorted(df['state'].unique())
state_data = {}
for state in states:
    state_df = df[df['state'] == state]
    bin_counts = state_df['duration_bin'].value_counts().reindex(labels, fill_value=0)
    state_data[state] = bin_counts.values.tolist()

init_state = states[0]
fig = go.Figure(data=[
    go.Bar(
        x=labels,
        y=state_data[init_state],
        name=init_state
    )
])

buttons = []
for state in states:
    buttons.append(dict(
        label=state,
        method="update",
        args=[{"y": [state_data[state]], "name": state},
              {"title": f"Stations Operating Duration for {state}",
               "yaxis": {"title": "Number of Stations"},
               "xaxis": {"title": "Operating Duration Interval"}}]
    ))

fig.update_layout(
    updatemenus=[dict(
        active=0,
        buttons=buttons,
        direction="down",
        x=0.0,
        xanchor="left",
        y=1.15,
        yanchor="top"
    )],
    title=f"Stations Operating Duration for {init_state}",
    xaxis_title="Operating Duration Interval",
    yaxis_title="Number of Stations"
)

fig.show()



