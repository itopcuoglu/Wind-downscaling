import numpy as np

Sandia_coords = [34.96226779999999, -106.5096873]

def find_closest_HRRR_loc(hrrr, hrrr_coords_to_use):
    
    """
    hrrr: HRRR xarray
    hrrr_coords_to_use: array -> [lat, lon]
    
    
    
    """

    dist = (hrrr['longitude'].values - hrrr_coords_to_use[1])**2 + (hrrr['latitude'].values - hrrr_coords_to_use[0])**2
    idx = np.unravel_index(np.argmin(dist), dist.shape)
    closest_lat = hrrr.latitude.values[idx]
    closest_lon = hrrr.longitude.values[idx]

    hrrr_loc =hrrr.sel(x=idx[1], y=idx[0])
    
    return hrrr_loc



def find_nearest_obs(obs, target_lat = Sandia_coords[0], target_lon = Sandia_coords[1] ):

    # Compute distance
    obs['distance_to_sandia'] = np.sqrt(
        (obs['lat'] - target_lat)**2 + (obs['lon'] - target_lon)**2
    )
    
    # Find the row with the smallest distance
    closest_station = np.unique(obs[obs.distance_to_sandia == np.nanmin(obs['distance_to_sandia'])].stid)[0]
    
    # Filter data for the closest station
    ts = obs[obs.stid == closest_station]

    return ts


def wspd_wdir_from_uv(u, v):
    """ 
        u   eastward wind component
        v   northward wind component
        wdir   direction where wind is coming from
    """
    
    wspd = (u**2 + v**2)**0.5
    wdir = np.degrees(np.arctan2(u, v)) +180

    return wspd, wdir

def uv_from_wspd_wdir(wspd, wdir):
    """ 
        u   eastward wind component
        v   northward wind component
        wdir   direction where wind is coming from, input in degrees
    """

    wind_dir_rad = np.radians(wdir)
    u = -wspd * np.sin(wind_dir_rad)
    v = -wspd * np.cos(wind_dir_rad)    

    return u,v

