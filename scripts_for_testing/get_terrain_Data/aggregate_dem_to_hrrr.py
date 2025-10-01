# Block-wise DEM-to-1km aggregation with overlap to avoid edge effects
def process_dem_in_blocks_to_1km(tif_dir, block_size_deg=2.0, overlap_deg=0.05):
    # List all DEM tiles and get their bounds
    tif_files = glob.glob(os.path.join(tif_dir, '*.tif'))
    bounds_list = []
    for fp in tif_files:
        with rasterio.open(fp) as src:
            b = src.bounds
            bounds_list.append((b.left, b.right, b.bottom, b.top))
    print(f"Found {len(tif_files)} DEM tiles in {tif_dir}")        
    # Compute global bounds
    min_lon = min(b[0] for b in bounds_list)
    max_lon = max(b[1] for b in bounds_list)
    min_lat = min(b[2] for b in bounds_list)
    max_lat = max(b[3] for b in bounds_list)
    # 1km resolution in degrees
    res_deg = 1.0 / 111.0
    # Block grid
    lon_blocks = np.arange(min_lon, max_lon, block_size_deg - overlap_deg)
    lat_blocks = np.arange(min_lat, max_lat, block_size_deg - overlap_deg)
    block_results = []
    block_transforms = []
    block_shapes = []
    block_lons = []
    block_lats = []
    total_blocks = len(lon_blocks) * len(lat_blocks)
    block_counter = 0
    for i, lon0 in enumerate(lon_blocks):
        for j, lat0 in enumerate(lat_blocks):
            block_counter += 1
            print(f"Processing block {block_counter}/{total_blocks} (lon: {lon0:.3f} to {min(lon0 + block_size_deg, max_lon):.3f}, lat: {lat0:.3f} to {min(lat0 + block_size_deg, max_lat):.3f})")
            lon1 = min(lon0 + block_size_deg, max_lon)
            lat1 = min(lat0 + block_size_deg, max_lat)
            # Expand by overlap
            lon0_ov = max(lon0 - overlap_deg, min_lon)
            lon1_ov = min(lon1 + overlap_deg, max_lon)
            lat0_ov = max(lat0 - overlap_deg, min_lat)
            lat1_ov = min(lat1 + overlap_deg, max_lat)
            bbox = (lon0_ov, lon1_ov, lat0_ov, lat1_ov)
            # Find overlapping DEM tiles
            filtered_files = []
            for fp, b in zip(tif_files, bounds_list):
                if not (b[1] < bbox[0] or b[0] > bbox[1] or b[3] < bbox[2] or b[2] > bbox[3]):
                    filtered_files.append(fp)
            if not filtered_files:
                print(f"  Skipping block {block_counter}: no DEM tiles overlap this block.")
                continue
            # Open DEMs in this block with per-file error handling
            srcs = []
            bad_files = []
            for fp in filtered_files:
                try:
                    srcs.append(rasterio.open(fp))
                except Exception as e:
                    print(f"  Skipping file (cannot open): {fp}\n    Error: {e}")
                    bad_files.append(fp)
            if not srcs:
                print(f"  Skipping block {block_counter}: all DEM files failed to open.")
                continue
            if bad_files:
                print(f"  Block {block_counter}: {len(bad_files)} files could not be opened and were skipped.")
            try:
                mosaic, out_trans = merge(srcs, bounds=(bbox[0], bbox[2], bbox[1], bbox[3]))
            except Exception as e:
                print(f"  Warning: Failed to merge files for block {block_counter}. Files: {[src.name for src in srcs]}\n  Error: {e}")
                for src in srcs:
                    src.close()
                continue
            for src in srcs:
                src.close()
            dem = mosaic[0]
            dem_profile = srcs[0].profile.copy()
            dem_profile.update({'transform': out_trans, 'height': dem.shape[0], 'width': dem.shape[1]})
            nodata = srcs[0].nodata
            # Resample to 1km grid for this block
            nx = int(np.ceil((lon1 - lon0) / res_deg))
            ny = int(np.ceil((lat1 - lat0) / res_deg))
            from affine import Affine
            dst_transform = Affine.translation(lon0, lat0) * Affine.scale(res_deg, res_deg)
            dst_shape = (ny, nx)
            from rasterio.warp import reproject, Resampling
            dem_resampled = np.full(dst_shape, np.nan, dtype=np.float32)
            reproject(
                source=dem,
                destination=dem_resampled,
                src_transform=out_trans,
                src_crs=dem_profile['crs'],
                dst_transform=dst_transform,
                dst_crs=dem_profile['crs'],
                resampling=Resampling.average,
                src_nodata=nodata,
                dst_nodata=np.nan,
                num_threads=2
            )
            dem_sq = dem ** 2
            dem_sq_resampled = np.full(dst_shape, np.nan, dtype=np.float32)
            reproject(
                source=dem_sq,
                destination=dem_sq_resampled,
                src_transform=out_trans,
                src_crs=dem_profile['crs'],
                dst_transform=dst_transform,
                dst_crs=dem_profile['crs'],
                resampling=Resampling.average,
                src_nodata=nodata,
                dst_nodata=np.nan,
                num_threads=2
            )
            std_resampled = np.sqrt(np.maximum(0, dem_sq_resampled - dem_resampled**2))
            # Store block
            block_results.append((dem_resampled, std_resampled))
            block_transforms.append(dst_transform)
            block_shapes.append(dst_shape)
            block_lons.append((lon0, lon1))
            block_lats.append((lat0, lat1))
    # Mosaic all 1km blocks
    # Compute global 1km grid
    global_min_lon = min(l[0] for l in block_lons)
    global_max_lon = max(l[1] for l in block_lons)
    global_min_lat = min(l[0] for l in block_lats)
    global_max_lat = max(l[1] for l in block_lats)
    nx = int(np.ceil((global_max_lon - global_min_lon) / res_deg))
    ny = int(np.ceil((global_max_lat - global_min_lat) / res_deg))
    global_transform = Affine.translation(global_min_lon, global_min_lat) * Affine.scale(res_deg, res_deg)
    global_mean = np.full((ny, nx), np.nan, dtype=np.float32)
    global_std = np.full((ny, nx), np.nan, dtype=np.float32)
    global_count = np.zeros((ny, nx), dtype=np.int32)
    # For each block, paste into global grid, averaging overlaps
    for (block_mean, block_std), t, shape, (lon0, lon1), (lat0, lat1) in zip(block_results, block_transforms, block_shapes, block_lons, block_lats):
        x0 = int(round((lon0 - global_min_lon) / res_deg))
        y0 = int(round((lat0 - global_min_lat) / res_deg))
        y1 = y0 + shape[0]
        x1 = x0 + shape[1]
        # For mean: running average
        block_mask = ~np.isnan(block_mean)
        global_mask = ~np.isnan(global_mean[y0:y1, x0:x1])
        # Where global is nan, just assign
        assign_mask = block_mask & ~global_mask
        global_mean[y0:y1, x0:x1][assign_mask] = block_mean[assign_mask]
        global_std[y0:y1, x0:x1][assign_mask] = block_std[assign_mask]
        global_count[y0:y1, x0:x1][assign_mask] = 1
        # Where both have data, average
        avg_mask = block_mask & global_mask
        global_mean[y0:y1, x0:x1][avg_mask] = (global_mean[y0:y1, x0:x1][avg_mask] * global_count[y0:y1, x0:x1][avg_mask] + block_mean[avg_mask]) / (global_count[y0:y1, x0:x1][avg_mask] + 1)
        global_std[y0:y1, x0:x1][avg_mask] = (global_std[y0:y1, x0:x1][avg_mask] * global_count[y0:y1, x0:x1][avg_mask] + block_std[avg_mask]) / (global_count[y0:y1, x0:x1][avg_mask] + 1)
        global_count[y0:y1, x0:x1][avg_mask] += 1
    # Build lat/lon arrays
    xs = np.arange(nx) * res_deg + global_min_lon + res_deg/2
    ys = np.arange(ny) * res_deg + global_min_lat + res_deg/2
    lon_grid, lat_grid = np.meshgrid(xs, ys)
    return global_mean, global_std, lat_grid, lon_grid
import os
import glob
import numpy as np
import rasterio
import xarray as xr
import matplotlib.pyplot as plt
from rasterio.merge import merge
from rasterio.warp import reproject, Resampling
import concurrent.futures
import time

"""
This script:

Reads and merges all 30m DEM tiles.
Loads the HRRR 3km grid from a NetCDF HRRR file.
For each 3km grid point, computes the mean and standard deviation of all 30m DEM cells within ±1.5 km.
Saves the results as an xarray NetCDF file (hrrr_elevation_3km.nc).
Plots the mean elevation on the HRRR grid.
"""



# Resample merged DEM to a 1km grid (ignore HRRR grid)
def resample_dem_to_1km_grid(dem, dem_transform, dem_profile, nodata):
    from affine import Affine
    from rasterio.warp import reproject, Resampling
    # Determine the bounds of the DEM
    left, top = dem_transform * (0, 0)
    right, bottom = dem_transform * (dem.shape[1], dem.shape[0])
    min_lon, max_lon = min(left, right), max(left, right)
    min_lat, max_lat = min(top, bottom), max(top, bottom)
    # 1km resolution in degrees (approx, at equator: 1 deg ~ 111km)
    res_deg = 1.0 / 111.0
    nx = int(np.ceil((max_lon - min_lon) / res_deg))
    ny = int(np.ceil((max_lat - min_lat) / res_deg))
    # Build 1km grid transform
    dst_transform = Affine.translation(min_lon, min_lat) * Affine.scale(res_deg, res_deg)
    dst_shape = (ny, nx)
    # Resample mean
    dem_resampled = np.full(dst_shape, np.nan, dtype=np.float32)
    reproject(
        source=dem,
        destination=dem_resampled,
        src_transform=dem_transform,
        src_crs=dem_profile['crs'],
        dst_transform=dst_transform,
        dst_crs=dem_profile['crs'],
        resampling=Resampling.average,
        src_nodata=nodata,
        dst_nodata=np.nan,
        num_threads=2
    )
    # Resample std
    dem_sq = dem ** 2
    dem_sq_resampled = np.full(dst_shape, np.nan, dtype=np.float32)
    reproject(
        source=dem_sq,
        destination=dem_sq_resampled,
        src_transform=dem_transform,
        src_crs=dem_profile['crs'],
        dst_transform=dst_transform,
        dst_crs=dem_profile['crs'],
        resampling=Resampling.average,
        src_nodata=nodata,
        dst_nodata=np.nan,
        num_threads=2
    )
    std_resampled = np.sqrt(np.maximum(0, dem_sq_resampled - dem_resampled**2))
    # Build lat/lon arrays
    xs = np.arange(nx) * res_deg + min_lon + res_deg/2
    ys = np.arange(ny) * res_deg + min_lat + res_deg/2
    lon_grid, lat_grid = np.meshgrid(xs, ys)
    return dem_resampled, std_resampled, lat_grid, lon_grid


def get_hrrr_grid(nc_path):
    ds = xr.open_dataset(nc_path)
    lat = ds['latitude'].values
    lon = ds['longitude'].values
    ds.close()
    # Convert HRRR longitudes from 0-360 to -180 to 180 if needed
    if np.nanmax(lon) > 180:
        lon = ((lon + 180) % 360) - 180
    return lat, lon

def read_and_merge_tiffs(tif_dir):
    # Accept optional bounding box for filtering
    def read_and_merge_tiffs_with_bbox(tif_dir, bbox=None):
        tif_files = glob.glob(os.path.join(tif_dir, '*.tif'))
        filtered_files = []
        print(f"Checking {len(tif_files)} DEM tiles for overlap with HRRR bbox...")
        for idx, fp in enumerate(tif_files):
            with rasterio.open(fp) as src:
                b = src.bounds
                # bbox = (min_lon, max_lon, min_lat, max_lat)
                # DEM tile bounds: b.left, b.right, b.bottom, b.top
                # Overlap if not (b.left > bbox[1] or b.right < bbox[0] or b.bottom > bbox[3] or b.top < bbox[2])
                if bbox is None or not (b.left > bbox[1] or b.right < bbox[0] or b.bottom > bbox[3] or b.top < bbox[2]):
                    filtered_files.append(fp)
            if (idx+1) % 500 == 0 or idx == len(tif_files)-1:
                print(f"  Checked {idx+1}/{len(tif_files)} tif files, {len(filtered_files)} overlap so far.")
        if not filtered_files:
            raise RuntimeError('No DEM tiles overlap the HRRR grid subset!')
        print(f"Merging {len(filtered_files)} DEM tiles...")
        src_files_to_mosaic = []
        for i, fp in enumerate(filtered_files):
            src_files_to_mosaic.append(rasterio.open(fp))
            if (i+1) % 100 == 0 or (i+1) == len(filtered_files):
                print(f"  Opened {i+1}/{len(filtered_files)} tiles for merging...")
        print("Starting rasterio.merge.merge...")
        mosaic, out_trans = merge(src_files_to_mosaic)
        print("DEM tiles merged.")
        profile = src_files_to_mosaic[0].profile.copy()
        profile.update({
            'height': mosaic.shape[1],
            'width': mosaic.shape[2],
            'transform': out_trans
        })
        nodata = src_files_to_mosaic[0].nodata
        for src in src_files_to_mosaic:
            src.close()
        print("DEM profile and nodata extracted.")
        return mosaic[0], out_trans, profile, nodata
    return read_and_merge_tiffs_with_bbox


# New fast resampling function using rasterio's reproject
def resample_dem_to_hrrr_grid_and_std(dem, dem_transform, dem_profile, hrrr_lat, hrrr_lon, nodata):
    from rasterio.warp import reproject, Resampling
    # Accept global HRRR grid transform and block position for correct alignment
    global_dst_transform = None
    global_x1 = None
    global_y1 = None
    global_nx = None
    global_ny = None
    import inspect
    frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues(frame)
    if 'global_dst_transform' in values and values['global_dst_transform'] is not None:
        global_dst_transform = values['global_dst_transform']
        global_x1 = values['global_x1']
        global_y1 = values['global_y1']
        global_nx = values['global_nx']
        global_ny = values['global_ny']
    ny, nx = hrrr_lat.shape
    # If global transform is provided, use it and extract the correct window for this block
    if global_dst_transform is not None and global_x1 is not None and global_y1 is not None and global_nx is not None and global_ny is not None:
        # The full HRRR grid shape is (global_ny, global_nx)
        # The block is at (global_y1:global_y1+ny, global_x1:global_x1+nx)
        dst_transform = global_dst_transform * Affine.translation(global_x1, global_y1)
        dst_shape = (ny, nx)
    else:
        # Fallback: old behavior (should not be used)
        lon1, lon2 = hrrr_lon[0,0], hrrr_lon[0,-1]
        lat1, lat2 = hrrr_lat[0,0], hrrr_lat[-1,0]
        xres = (lon2 - lon1) / (nx - 1)
        yres = (lat2 - lat1) / (ny - 1)
        from affine import Affine
        dst_transform = Affine.translation(lon1 - xres/2, lat1 - yres/2) * Affine.scale(xres, yres)
        dst_shape = (ny, nx)
    # Progress callback for mean and std
    def progress_callback(complete, total, label):
        pct = 100 * complete / total
        if complete % max(1, total // 10) == 0 or complete == total:
            print(f"  {label}: {complete}/{total} ({pct:.1f}%) HRRR grid points processed.")

    # Mean
    dem_resampled = np.full(dst_shape, np.nan, dtype=np.float32)
    print("Starting rasterio.warp.reproject for mean ...")
    reproject(
        source=dem,
        destination=dem_resampled,
        src_transform=dem_transform,
        src_crs=dem_profile['crs'],
        dst_transform=dst_transform,
        dst_crs=dem_profile['crs'],
        resampling=Resampling.average,
        src_nodata=nodata,
        dst_nodata=np.nan,
        num_threads=2,
        progress_callback=lambda c, t: progress_callback(c, t, 'mean')
    )
    # Mean of squares
    dem_sq = dem ** 2
    dem_sq_resampled = np.full(dst_shape, np.nan, dtype=np.float32)
    print("Starting rasterio.warp.reproject for std ...")
    reproject(
        source=dem_sq,
        destination=dem_sq_resampled,
        src_transform=dem_transform,
        src_crs=dem_profile['crs'],
        dst_transform=dst_transform,
        dst_crs=dem_profile['crs'],
        resampling=Resampling.average,
        src_nodata=nodata,
        dst_nodata=np.nan,
        num_threads=2,
        progress_callback=lambda c, t: progress_callback(c, t, 'std')
    )
    # Std
    std_resampled = np.sqrt(np.maximum(0, dem_sq_resampled - dem_resampled**2))
    return dem_resampled, std_resampled



if __name__ == '__main__':
    tif_dir = os.path.join(os.path.dirname(__file__), 'CONUS_terrain_files')
    print('Processing DEM in blocks to 1km grid...')
    mean_elev, std_elev, lat_grid, lon_grid = process_dem_in_blocks_to_1km(tif_dir)
    print('Saving to NetCDF...')
    ds = xr.Dataset({
        'elevation_mean': (('y', 'x'), mean_elev),
        'elevation_std': (('y', 'x'), std_elev),
        'lat': (('y', 'x'), lat_grid),
        'lon': (('y', 'x'), lon_grid)
    })
    # ds.to_netcdf('/kfs2/projects/sfcwinds/environmental_data/CONUS_elevation_1km.nc')
    print('Plotting mean elevation...')
    plt.figure(figsize=(10,6))
    plt.pcolormesh(lon_grid, lat_grid, mean_elev, cmap='terrain')
    plt.colorbar(label='Mean Elevation (m)')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Mean Elevation on 1km Grid')
    plt.tight_layout()
    plt.show()