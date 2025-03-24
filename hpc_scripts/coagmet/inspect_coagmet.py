import pandas as pd

file_path = "/projects/sfcwinds/data/CO/coagmet_5min_2010_2025.csv"

df = pd.read_csv(file_path, nrows=10)

print("\nDataset Info:")
print(df.info())

print("\nSample Data:")
print(df.head())

print("\nMissing Values per Column:")
print(df.isna().sum())

print("\nColumn Names:")
print(df.columns)




