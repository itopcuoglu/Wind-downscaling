
import os
import requests

def download_files(file_list_path, output_dir, num_files=5):
	# Ensure output directory exists
	os.makedirs(output_dir, exist_ok=True)
	# Read the first num_files URLs
	with open(file_list_path, 'r') as f:
		urls = [line.strip() for _, line in zip(range(num_files), f)]
	for url in urls:
		if not url:
			continue
		filename = os.path.join(output_dir, os.path.basename(url))
		# print(f"Downloading {url} to {filename} ...")
		try:
			response = requests.get(url, stream=True)
			response.raise_for_status()
			with open(filename, 'wb') as out_file:
				for chunk in response.iter_content(chunk_size=8192):
					out_file.write(chunk)
			print(f"Downloaded: {filename}")
		except Exception as e:
			print(f"Failed to download {url}: {e}")

if __name__ == "__main__":
	file_list = os.path.join(os.path.dirname(__file__), "terrain_file_list.txt")
	output_dir = os.path.join(os.path.dirname(__file__), "CONUS_terrain_files")
	download_files(file_list, output_dir, num_files=50000000)
