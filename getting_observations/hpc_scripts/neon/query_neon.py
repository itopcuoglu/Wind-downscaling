import requests

dataset_id = "DP1.00001.001"  # NEON wind dataset
url = f"https://data.neonscience.org/api/v0/products/{dataset_id}"

response = requests.get(url)

if response.status_code == 200:
    data = response.json()
    
    print("Connection successful")
    print("Available Data Files:")

    available_sites = data.get("data", {}).get("siteCodes", [])
    if available_sites:
        for site in available_sites:
            site_code = site["siteCode"]
            months = site.get("availableMonths", [])
            print(f"Site: {site_code} -> Available Months: {len(months)} months of data")
    else:
        print("No available site data found.")

else:
    print(f"API request failed with status code {response.status_code}")
    print(response.text)



