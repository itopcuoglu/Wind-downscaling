import os
import requests

download_list = "/projects/sfcwinds/data/neon/neon_2min_download_links.txt"
output_dir = "/projects/sfcwinds/data/neon_raw/"

os.makedirs(output_dir, exist_ok=True)

with open(download_list, "r") as f:
    lines = f.readlines()

skipped = 0
downloaded = 0

for line in lines:
    parts = line.strip().split(",")
    if len(parts) < 4:
        continue  # Skip invalid lines

    site, json_file, filename, url = parts

    site_dir = os.path.join(output_dir, site)
    os.makedirs(site_dir, exist_ok=True)

    year = json_file.split("-")[0]  # Extract year from JSON filename
    year_dir = os.path.join(site_dir, year)
    os.makedirs(year_dir, exist_ok=True)

    file_path = os.path.join(year_dir, filename)

    if os.path.exists(file_path):
        skipped += 1
        continue

    print(f"⬇️ Downloading {filename} → {file_path}")
    response = requests.get(url, stream=True)

    if response.status_code == 200:
        with open(file_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        downloaded += 1
        print(f"Downloaded: {filename}")
    else:
        print(f"Failed: {filename} (HTTP {response.status_code})")

print(f"\nDownload Complete")

