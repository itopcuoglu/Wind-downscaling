import os
import requests
from tqdm import tqdm

BASE_URL = "https://www.ncei.noaa.gov/pub/data/uscrn/products/subhourly01/"
TARGET_DIR = "/projects/sfcwinds/data/USCRN/"

START_YEAR = 2014
END_YEAR = 2025

os.makedirs(TARGET_DIR, exist_ok=True)

def download_file(url, save_path):
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get("content-length", 0))
        with open(save_path, "wb") as file, tqdm(
            desc=save_path, total=total_size, unit="B", unit_scale=True, unit_divisor=1024
        ) as bar:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    file.write(chunk)
                    bar.update(len(chunk))
        return True
    except requests.RequestException as e:
        print(f"Failed to download {url}: {e}")
        return False

for year in range(START_YEAR, END_YEAR + 1):
    year_dir = os.path.join(TARGET_DIR, str(year))
    os.makedirs(year_dir, exist_ok=True)

    year_url = f"{BASE_URL}{year}/"
    station_list = requests.get(year_url).text.split("\n")

    for line in station_list:
        if 'CRNS0101-05' in line and '.txt' in line:
            file_name = line.split('">')[1].split("</a>")[0]  # Extract filename
            file_url = f"{year_url}{file_name}"
            save_path = os.path.join(year_dir, file_name)

            if os.path.exists(save_path):
                print(f"✔Already downloaded: {file_name}")
                continue

            print(f"⬇ Downloading {file_name}...")
            download_file(file_url, save_path)

print("Download complete")



