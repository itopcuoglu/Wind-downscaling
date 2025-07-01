import os
import requests

output_dir = "/projects/sfcwinds/data/CO/"
os.makedirs(output_dir, exist_ok=True)

base_url = "https://coagmet.colostate.edu/rawdata/"

stations = [
    "AKR02", "ALT01", "AVN01", "BLA01", "BNV01", "BRG01", "BRK01", "BRL01",
    "BRL02", "BRL03", "CBL01", "CBN01", "CCR01", "CDG01", "CHT01", "CKP01",
    "CLK01", "CNN01", "COW01", "CTR01", "CTR02", "CTZ01", "CYA01", "DEN01",
    "DLR01", "DLT01", "DRG01", "DVC01", "EAC01", "EGL01", "EKT01", "FCC01",
    "FCL01", "FRT01", "FRT02", "FRT03", "FTC01", "FTC03", "FTL01", "FTM01",
    "FWL01", "GBY01", "GJC01", "GLY03", "GLY04", "GUN01", "GYP01", "HEB01",
    "HLY01", "HLY02", "HNE01", "HOT01", "HOT02", "HRT01", "HXT01", "HYD01",
    "HYK02", "IDL01", "IGN01", "IGN02", "ILF01", "JFN01", "KLN01", "KRK01",
    "KRM01", "KSY01", "KSY02", "LAM01", "LAM02", "LAM03", "LAM04", "LAR01",
    "LBN01", "LCN01", "LJR01", "LJT01", "LSL01", "LMS02", "MCL01", "MKR01",
    "MNC01", "MTR01", "NUC01", "NWD01", "ORM01", "ORM02", "OTH01", "OTH02",
    "PAI01", "PAN01", "PBL01", "PBW01", "PGS01", "PKH01", "PKN01", "PKR01",
    "PNR01", "PTV01", "RFD01", "RFD02", "SAN01", "SBT01", "SCM01", "SLD01",
    "SLT01", "STG01", "STN01", "STT01", "TWC01", "UWR70", "VLD01", "WAV01",
    "WCF01", "WFD01", "WGG01", "WGG02", "WLS01", "WLT01", "WRY01", "WRY02",
    "YJK01", "YUC01", "YUM01", "YUM02"
]

def download_station_data(station_id):
    url = f"{base_url}{station_id}.dat"
    response = requests.get(url)
    if response.status_code == 200:
        file_path = os.path.join(output_dir, f"{station_id}.dat")
        with open(file_path, 'wb') as file:
            file.write(response.content)
        print(f"Downloaded data for station {station_id}")
    else:
        print(f"Failed to download data for station {station_id}")

for station in stations:
    download_station_data(station)




