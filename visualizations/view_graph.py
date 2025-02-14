import os
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
file_path = os.path.join(BASE_DIR, "..", "data", "stations.csv") 

df = pd.read_csv(file_path)

try:
    df = pd.read_csv(file_path)

    if 'height' in df.columns:
        heights = df['height'].dropna().astype(float)
    else:
        raise ValueError("'height' column not found in stations.csv")

    height_counts = heights.value_counts().sort_index()

    plt.figure(figsize=(10, 6))
    plt.bar(height_counts.index.astype(str), height_counts.values, color='dodgerblue')

    plt.xlabel("Measurement Height (m)")
    plt.ylabel("Number of Stations")
    plt.title("Number of Weather Stations at Each Exact Height")
    plt.xticks(rotation=45)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    plt.show()

except FileNotFoundError:
    print(f"Error: {file_path} not found")

except Exception as e:
    print(f"Unexpected Error: {e}")



