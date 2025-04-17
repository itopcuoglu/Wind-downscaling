# Wind Downscaling Using Observational Weather Stations

This project develops a high-resolution, near-surface wind observation database for the U.S., with a focus on the Southwest. It integrates wind observations from a wide range of station networks (e.g., AZMet, CoAgMet, NEON, AmeriFlux) and prepares the data for machine learning-based downscaling of model forecasts (e.g., HRRR).

The goal is to support improved wind forecasting at heights relevant to solar infrastructure and grid resilience planning.

## Project Structure

- `scripts/`: Main database and processing pipeline scripts.
  - `create_database.py`: Initializes the observation database.
  - `add_station_metadata.py`: Adds station metadata to the database.
  - `add_observations.py`: Loads, cleans, and standardizes observation data into Parquet format.
  - `view_db.py`: Tool for inspecting the database contents.
  - `view_parquet.py`: Quick preview of Parquet data.

- `hpc_scripts/`: Network-specific processing code designed for HPC use.
  - Subfolders (e.g., `azmet/`, `coagmet/`, `neon/`, etc.) each contain data-specific formatting scripts.
  - `conversion_template.py`: Template used across networks to format observations to a shared schema.

- `synoptic_data/`: Scripts and station lists for accessing data via the Synoptic (MesoWest) API.
  - `download_synoptic_data.py`: Pulls observational data from the Synoptic API.
  - `Read_Synoptic_downloaded_data.py`: Parses downloaded Synoptic JSON files.
  - `US_stations_under_10m.csv`: Curated list of U.S. Synoptic stations with measurement height ≤10m.
  - `NM_stations_under_10m.csv`: Subset for New Mexico.

- `download_model_data/`: Scripts for acquiring HRRR model output.
  - `Download_HRRR_data.py`: Retrieves HRRR wind fields.
  - `Convert_HRRR_grid.py`: Converts gridded data into usable input format for ML.

- `download_obs_data/`: Script for acquiring METAR weather station observations.
  - `Download_METAR_data.py`: Pulls and formats METAR hourly wind data.

- `visualizations/`: Data visualization and quality control tools.
  - `view_station_map.py`: Generates an interactive U.S. map of station locations.
  - `data_overview.py`: Builds a dashboard showing state coverage, network distributions, and summary metrics.
  - `view_graph.py`, `view_station_timeline.py`: Additional plotting tools.

- `toy_problem/`: Toy example testing wind downscaling on a small dataset using a Random Forest model.

- `ML_model_tests/`: Early experiments with model architectures (e.g., CNN). Not currently active.

- `data/`: Metadata files used in the database build process.
  - `stations.csv`: Master station metadata file.

- `requirements.txt`: Required Python packages.
- `README.md`: Project documentation.

## Setup

1. **Create a virtual environment**:
   ```bash
   python -m venv venv

2. **Activate the virtual environment**:
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

3. **Install the required packages**:
   ```bash
   pip install -r requirements.txt

## Running the Scripts

This project is not intended for reuse as a public package. However, here is the general workflow used during development:

1. **Database Setup**:
   ```bash
   python scripts/create_database.py
   python scripts/add_station_metadata.py
   python scripts/add_observations.py

2. **Visualization and Summary Tools**
   ```bash
   python visualizations/view_station_map.py
   python visualizations/data_overview.py

3. **HRRR Model Data:**
   ```bash
   python download_model_data/Download_HRRR_data.py
   python download_model_data/Convert_HRRR_grid.py

4. **Synoptic API Downloads:**
   ```bash
   python synoptic_data/download_synoptic_data.py

## Visualization Scripts

### `view_station_map.py`
Generates an interactive U.S. map showing the geographic distribution of all weather stations in the database. Stations are color-coded by state, and hover text displays metadata including elevation, lat/lon, and network.

### `data_overview.py`
Builds a dashboard with multiple plots:
- U.S. station map (CONUS only)
- Bar chart of station count per state
- Pie chart of network distribution (labels removed)
- Histogram of operating durations
- Summary table with average height, elevation, and number of stations

This script is useful for getting a quick overview of the coverage and quality of the dataset.

### `view_graph.py`
Generates time series plots for a selected station and variable. Useful for inspecting the consistency or anomalies in raw observations.

### `view_station_timeline.py`
Creates a horizontal bar chart showing the operating periods (start/end) for each station in a selected network.

## Outputs

Observation data are exported in Parquet format and stored externally (typically on HPC):
outputs/<NETWORK>/<STATION_ID>/<YEAR>.parquet

Each `.parquet` file contains the following standardized columns:
- `timestamp`
- `windspeed`
- `winddirection`
- `gust`
- `temperature`
- `qualitycontrol`

These files are partitioned by both station and year for efficient processing.

## Data Sources

This project incorporates observational data from the following networks:

- **AZMet** – Arizona Meteorological Network: https://azmet.arizona.edu/
- **CoAgMet** – Colorado Agricultural Meteorological Network: https://coagmet.colostate.edu/  
- **NEON** – National Ecological Observatory Network: https://data.neonscience.org/data-products/DP1.00001.001
- **AmeriFlux** – Flux tower network: https://ameriflux.lbl.gov/
- **USCRN** – U.S. Climate Reference Network: https://cds.climate.copernicus.eu/datasets/insitu-observations-near-surface-temperature-us-climate-reference-network?tab=overview  
- **Synoptic (via MesoWest API)** – Aggregated station data: https://synopticdata.com/  
- **HRRR** – High-Resolution Rapid Refresh model from NOAA  
- **METAR** – Hourly airport observations for select sites

## Requirements

This project requires the following Python packages:
- pandas
- numpy
- matplotlib
- plotly
- seaborn
- scikit-learn
- xarray
- pygrib
- pyproj
- sqlite3

Install them using:
```bash
pip install -r requirements.txt
