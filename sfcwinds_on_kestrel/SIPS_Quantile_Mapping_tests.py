import os
import pandas as pd
import xarray as xr
import glob
#matplotlib widget
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow
from datetime import datetime
import numpy as np
from scipy.stats import norm, gamma, erlang, expon
import matplotlib.dates as mdates
from sklearn.metrics import root_mean_squared_error

from Functions_sfcwinds import *


# Enable interactive backend when running in IPython, ignore in plain Python scripts
try:
    if callable(globals().get("get_ipython", None)):
        get_ipython().run_line_magic("matplotlib", "widget")  # type: ignore[name-defined]
except Exception:
    pass



#%% Read data
# Read metadata to filter what we want
meta = pd.read_csv("/kfs2/projects/sfcwinds/observations/metadata_CONUS.csv")

# filter which data to read
base_dir = "/kfs2/projects/sfcwinds/test_case/Quantile mapping May 2024/"

stid = "SVAN5"   

# Read files from base_dir
merged = pd.read_csv(os.path.join(base_dir, f"{stid}_3m_hrrr_10m_merged.csv"))
quantile_10m_data = pd.read_csv(os.path.join(base_dir, f"{stid}_3m_hrrr_10m_quantile_data.csv"))
quantile_10m_stats = pd.read_csv(os.path.join(base_dir, f"{stid}_3m_hrrr_10m_quantile_stats.csv"))





#%% Plot original data


# Wspd time series
fig, ax = plt.subplots()
ax.plot(merged.valid_time[~np.isnan(merged.windspeed_3m)],merged.windspeed_3m[~np.isnan(merged.windspeed_3m)],label= f"{stid}"' 3m')
ax.plot(merged.valid_time[~np.isnan(merged.windspeed_3m)],merged.wspd10[~np.isnan(merged.windspeed_3m)],label='HRRR 10m')
ax.legend(loc='upper right',fontsize=10)
ax.set_xlabel('Date')
ax.set_ylabel('Wind speed (m/s)')
ax.legend(loc='upper right',fontsize=10)
fig.autofmt_xdate()
#fig.savefig(f"{stid} 3m vs hrrr 10m wind speed test case NM May 2024.png")
#plt.tight_layout()
#plt.savefig("obs num time resolution NM.png")
#plt.xlim([datetime.datetime(2024,8,26,0,0), datetime.datetime(2024,9,10,0,0)])
#plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))

# Wdir time series
fig, ax = plt.subplots()
ax.plot(merged.valid_time[~np.isnan(merged.winddirection)],merged.winddirection[~np.isnan(merged.winddirection)],label= f"{stid}"' 3m')
ax.plot(merged.valid_time[~np.isnan(merged.winddirection)],merged.wdir10[~np.isnan(merged.winddirection)],label='HRRR 10m')
ax.legend(loc='upper right',fontsize=10)
ax.set_xlabel('Date')
ax.set_ylabel('Wind direction (deg)')
ax.legend(loc='upper right',fontsize=10)
fig.autofmt_xdate()
# fig.savefig(f"{stid} 3m  hrrr 10m wind direction test case NM May 2024.png")
    

# Wsdp histogram
plt.figure()
hist1 = plt.hist(merged.windspeed_3m, bins=np.arange(0, 16, 0.1), edgecolor='none',
                 density=False,alpha=0.5,label= f"{stid}"' 3m')
hist2 = plt.hist(merged.wspd10, bins=np.arange(0, 16, 0.1), edgecolor='none',
                 density=False,alpha=0.5,label='hrrr 10m')
plt.xlabel("Wind speed (m/s)")
plt.ylabel("Frequency")
plt.legend(loc='upper right',fontsize=10)
plt.savefig(f"{stid} 3m vs hrrr 10m wind speed histogram test case NM May 2024.png")
plt.show()


#%% Wspd quantile mapping

# Define number of quantiles
q_ = 50 

# Wind speed time series
obs_windspeed_3m = merged.windspeed_3m[~np.isnan(merged.windspeed_3m)]
hrrr_windspeed_10m = merged.wspd10[~np.isnan(merged.windspeed_3m)]


### Quantiles based on normal distribution (parametric)

# quantiles = [round(x*0.05,2) for x in range(1,q_)]      # for 5% increments up to 95%

# dist_obs_ws = norm(loc=obs_windspeed_3m.mean(), scale=obs_windspeed_3m.std())
# dist_hrrr_ws = norm(loc=hrrr_windspeed_10m.mean(), scale=hrrr_windspeed_10m.std())

# q_dist_obs_ws = [dist_obs_ws.ppf(x*0.05) for x in range(1,q_)]
# q_dist_hrrr_ws = [dist_hrrr_ws.ppf(x*0.05) for x in range(1,q_)]
# q_dist_diff_ws = np.array(q_dist_obs_ws)-np.array(q_dist_hrrr_ws)

# #Quantiles Obs and HRRR
# plt.figure()
# plt.plot(quantiles,q_dist_obs_ws,alpha=0.5,label= f"{stid}")
# plt.plot(quantiles,q_dist_hrrr_ws,alpha=0.5,label='hrrr')
# plt.xlabel("Quantiles")
# plt.ylabel("Values")
# plt.xticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
# plt.legend(loc='upper left',fontsize=10)
# plt.savefig(f"{stid} 3m vs hrrr 10m wind speed quantiles test case NM May 2024.png")
# plt.show()



### Empirical quantiles (non-parametric CDF)
quantiles = np.arange(0,1+1/q_,1/q_)
quantiles_obs_ws_emp = np.quantile(obs_windspeed_3m, quantiles)
quantiles_hrrr_ws_emp = np.quantile(hrrr_windspeed_10m, quantiles)
quantiles_diff_ws_emp = quantiles_obs_ws_emp - quantiles_hrrr_ws_emp

# CDFs Obs and HRRR
plt.figure()
plt.plot(quantiles,quantiles_obs_ws_emp,alpha=0.5,label= f"{stid}")
plt.plot(quantiles,quantiles_hrrr_ws_emp,alpha=0.5,label='hrrr')
plt.xlabel("Quantiles")
plt.ylabel("Values")
plt.xticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
plt.legend(loc='upper left',fontsize=10)
plt.savefig(f"{stid} 3m vs hrrr 10m wind speed quantiles test case NM May 2024.png")
plt.show()

"""
Test this later: mapping to new stations

from scipy.stats import rankdata
from sklearn.ensemble import RandomForestRegressor

def build_quantile_mapping_pairs(model, obs, quantiles):
    # Create mapping pairs for one station.
    model_q = np.quantile(model, quantiles)
    obs_q = np.quantile(obs, quantiles)
    return model_q, obs_q

# -------------------------------
# Step 1: Build training dataset
# -------------------------------
all_records = []

quantiles = np.linspace(0.01, 0.99, 99)

for station_id, (model_train, obs_train, features) in enumerate(training_stations):
    # features is a dict: {"lat": ..., "elev": ..., ...}
    model_q, obs_q = build_quantile_mapping_pairs(model_train, obs_train, quantiles)

    for p, mq, oq in zip(quantiles, model_q, obs_q):
        rec = {
            "station_id": station_id,
            "quantile": p,
            "model_q": mq,
            "obs_q": oq,
            **features  # unpack station metadata
        }
        all_records.append(rec)

df = pd.DataFrame(all_records)

# -------------------------------
# Step 2: Train ML model
# -------------------------------
X = df.drop(columns=["station_id", "obs_q"])
y = df["obs_q"]

ml_model = RandomForestRegressor(n_estimators=200, random_state=0)
ml_model.fit(X, y)

# -------------------------------
# Step 3: Apply to unseen station
# -------------------------------
# new station with only model data + metadata
new_model = unseen_model_data
new_features = {"lat": 35.0, "elev": 1500, "dist_coast": 40}

# construct quantiles for unseen station
new_model_q = np.quantile(new_model, quantiles)

# build prediction dataframe
test_records = []
for p, mq in zip(quantiles, new_model_q):
    rec = {
        "quantile": p,
        "model_q": mq,
        **new_features
    }
    test_records.append(rec)

X_new = pd.DataFrame(test_records)

# predict "artificial observed quantiles"
predicted_obs_q = ml_model.predict(X_new)

# this gives the artificial station CDF
# -> map new model values via interpolation
mapped_values = np.interp(new_model, new_model_q, predicted_obs_q)
"""


# Reconstruct the time series based on quantile mapping

def quantile_mapping(model_series, model_cdf, obs_cdf, quantiles):
    # Find the quantile of each model value in the model CDF
    Quantiles = np.interp(model_series, model_cdf, quantiles)
    # Map these quantiles to observed values using the obs CDF
    corrected_series = np.interp(Quantiles, quantiles, obs_cdf)
    return corrected_series

corrected_hrrr_windspeed = quantile_mapping(
    hrrr_windspeed_10m,
    quantiles_hrrr_ws_emp,
    quantiles_obs_ws_emp,
    quantiles
)

# Time series plot of original and corrected model data
fig, ax = plt.subplots()
ax.plot(merged.valid_time[~np.isnan(merged.windspeed_3m)], obs_windspeed_3m, label=f"{stid} 3m (obs)")
ax.plot(merged.valid_time[~np.isnan(merged.windspeed_3m)], hrrr_windspeed_10m, label="HRRR 10m (model)")
ax.plot(merged.valid_time[~np.isnan(merged.windspeed_3m)], corrected_hrrr_windspeed, label="Corrected 10m (quantile-mapped)")
ax.set_xlabel("Date")
ax.set_ylabel("Wind speed (m/s)")
ax.legend(loc="upper right", fontsize=10)
ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=8))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
fig.autofmt_xdate()
plt.title("Observed, Model, and Corrected Wind Speed Time Series")
plt.show()

# Histogram of wind speed values for obs, model, and corrected
plt.figure()
plt.hist(obs_windspeed_3m, bins=np.arange(0, 16, 0.2), alpha=0.5, label=f"{stid} 3m (obs)", edgecolor='black')
plt.hist(hrrr_windspeed_10m, bins=np.arange(0, 16, 0.2), alpha=0.5, label="HRRR 10m (model)", edgecolor='black')
plt.hist(corrected_hrrr_windspeed, bins=np.arange(0, 16, 0.2), alpha=0.5, label="Corrected 10m (quantile-mapped)", edgecolor='black')
plt.xlabel("Wind speed (m/s)")
plt.ylabel("Frequency")
plt.legend(loc="upper right", fontsize=10)
plt.title("Histogram of Wind Speed: Obs, Model, Corrected")
plt.show()


# Correlation scatter plot: obs vs. original model and obs vs. corrected model

# Calculate metrics for original model
mask = ~np.isnan(obs_windspeed_3m) & ~np.isnan(hrrr_windspeed_10m)
r_model = np.corrcoef(obs_windspeed_3m[mask], hrrr_windspeed_10m[mask])[0,1]
rmse_model = root_mean_squared_error(obs_windspeed_3m[mask], hrrr_windspeed_10m[mask])
bias_model = np.mean(hrrr_windspeed_10m[mask] - obs_windspeed_3m[mask])

# Calculate metrics for corrected model
mask_corr = ~np.isnan(obs_windspeed_3m) & ~np.isnan(corrected_hrrr_windspeed)
r_corr = np.corrcoef(obs_windspeed_3m[mask_corr], corrected_hrrr_windspeed[mask_corr])[0,1]
rmse_corr = root_mean_squared_error(obs_windspeed_3m[mask_corr], corrected_hrrr_windspeed[mask_corr])
bias_corr = np.mean(corrected_hrrr_windspeed[mask_corr] - obs_windspeed_3m[mask_corr])

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.scatter(obs_windspeed_3m, hrrr_windspeed_10m, alpha=0.5, label=f"r = {r_model:.2f}\nRMSE = {rmse_model:.2f}\nBias = {bias_model:.2f}")
plt.xlabel(f"{stid} 3m (obs)")
plt.ylabel("HRRR 10m (model)")
plt.title("Obs vs. Original Model")
plt.grid(True)
plt.legend(loc='upper left', fontsize=10)

plt.subplot(1,2,2)
plt.scatter(obs_windspeed_3m, corrected_hrrr_windspeed, alpha=0.5, label=f"r = {r_corr:.2f}\nRMSE = {rmse_corr:.2f}\nBias = {bias_corr:.2f}")
plt.xlabel(f"{stid} 3m (obs)")
plt.ylabel("Corrected 10m (quantile-mapped)")
plt.title("Obs vs. Corrected Model")
plt.grid(True)
plt.legend(loc='upper left', fontsize=10)

plt.tight_layout()
plt.show()


