import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import os
from matplotlib.widgets import Button

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(BASE_DIR, "..", "data", "stations.csv")

df = pd.read_csv(file_path)

df["begints"] = pd.to_datetime(df["begints"], errors="coerce")
df["endts"] = pd.to_datetime(df["endts"], errors="coerce")

df["endts"] = df["endts"].fillna(pd.Timestamp.today())

df["display_name"] = df["station_name"] + " (" + df["state"] + ")"

df = df.sort_values("begints")

STATIONS_PER_PAGE = 10
num_stations = len(df)
num_pages = (num_stations // STATIONS_PER_PAGE) + (num_stations % STATIONS_PER_PAGE > 0)

X_START = dt.datetime(1980, 1, 1)
X_END = pd.Timestamp.today() 

fig, ax = plt.subplots(figsize=(12, 6))
plt.subplots_adjust(bottom=0.2)  


def plot_page(page):
    ax.clear()
    start_idx = page * STATIONS_PER_PAGE
    end_idx = min(start_idx + STATIONS_PER_PAGE, num_stations)

    for i, row in df.iloc[start_idx:end_idx].iterrows():
        ax.barh(row["display_name"], (row["endts"] - row["begints"]).days,
                left=row["begints"], color='dodgerblue', alpha=0.75)

    ax.set_xlabel("Year")
    ax.set_ylabel("Station (State)")
    ax.set_title(f"Operational Period of Weather Stations (Page {page+1}/{num_pages})")

    ax.set_xlim(X_START, X_END)

    ax.xaxis.set_major_locator(plt.MultipleLocator(365*4))  
    ax.set_xticks(pd.date_range(X_START, X_END, freq='4Y'))
    ax.set_xticklabels([str(year.year) for year in pd.date_range(X_START, X_END, freq='4Y')])

    ax.axvline(x=X_END, color='red', linestyle='--', label="Current")

    plt.grid(axis="x", linestyle="--", alpha=0.5)
    plt.legend(loc='upper left')

    fig.canvas.draw()


current_page = 0


def next_page(event):
    global current_page
    if current_page < num_pages - 1:
        current_page += 1
        plot_page(current_page)


def prev_page(event):
    global current_page
    if current_page > 0:
        current_page -= 1
        plot_page(current_page)


axprev = plt.axes([0.7, 0.05, 0.1, 0.075])  
axnext = plt.axes([0.81, 0.05, 0.1, 0.075])  
bnext = Button(axnext, 'Next')
bprev = Button(axprev, 'Previous')

bnext.on_clicked(next_page)
bprev.on_clicked(prev_page)

plot_page(current_page)
plt.show()



