import requests
import os
import re
from bs4 import BeautifulSoup
from urllib.parse import urljoin

STATION_URLS = {
    "07": "https://cales.arizona.edu/azmet/07.htm",
    "09": "https://cales.arizona.edu/azmet/09.htm",
    "33": "https://cales.arizona.edu/azmet/33.htm",
    "26": "https://cales.arizona.edu/azmet/26.htm",
    "05": "https://cales.arizona.edu/azmet/05.htm",
    "27": "https://cales.arizona.edu/azmet/27.htm",
    "28": "https://cales.arizona.edu/azmet/28.htm",
    "20": "https://cales.arizona.edu/azmet/20.htm",
    "06": "https://cales.arizona.edu/azmet/06.htm",
    "23": "https://cales.arizona.edu/azmet/23.htm",
    "40": "https://cales.arizona.edu/azmet/40.htm",
    "19": "https://cales.arizona.edu/azmet/19.htm",
    "08": "https://cales.arizona.edu/azmet/08.htm",
    "35": "https://cales.arizona.edu/azmet/35.htm",
    "32": "https://cales.arizona.edu/azmet/32.htm",
    "15": "https://cales.arizona.edu/azmet/15.htm",
    "04": "https://cales.arizona.edu/azmet/04.htm",
    "24": "https://cales.arizona.edu/azmet/24.htm",
    "22": "https://cales.arizona.edu/azmet/22.htm",
    "12": "https://cales.arizona.edu/azmet/12.htm",
    "39": "https://cales.arizona.edu/azmet/39.htm",
    "01": "https://cales.arizona.edu/azmet/01.htm",
    "37": "https://cales.arizona.edu/azmet/37.htm",
    "41": "https://cales.arizona.edu/azmet/41.htm",
    "38": "https://cales.arizona.edu/azmet/38.htm",
    "02": "https://cales.arizona.edu/azmet/02.htm",
    "36": "https://cales.arizona.edu/azmet/36.htm",
    "14": "https://cales.arizona.edu/azmet/14.htm"
}

OUTPUT_DIR = "/projects/sfcwinds/outputs/AZMet/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def fetch_hourly_data_links(station_id, station_url):
    try:
        response = requests.get(station_url)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        data_links = []
        for link in soup.find_all('a', href=True):
            match = re.search(r'(\d{2})(\d{2})rh\.txt$', link['href'])
            if match and match.group(1) == station_id:  
                full_url = urljoin(station_url, link['href'])
                data_links.append(full_url)
        
        return data_links
    except Exception as e:
        print(f"Error fetching data from {station_url}: {e}")
        return []

all_hourly_links = []
for station_id, station_url in STATION_URLS.items():
    hourly_data_links = fetch_hourly_data_links(station_id, station_url)
    all_hourly_links.extend(hourly_data_links)

print(f"Found {len(all_hourly_links)} hourly data files.")

def download_data(file_url):
    try:
        filename = os.path.basename(file_url)
        output_path = os.path.join(OUTPUT_DIR, filename)
        
        response = requests.get(file_url, stream=True)
        if response.status_code == 200:
            with open(output_path, 'wb') as file:
                for chunk in response.iter_content(chunk_size=1024):
                    file.write(chunk)
            print(f"Downloaded: {filename}")
        else:
            print(f"Failed to download: {file_url}")
    except Exception as e:
        print(f"Error downloading {file_url}: {e}")

for url in all_hourly_links:
    download_data(url)

print("AZMet hourly data download complete")



