from scripts import add_observations
from scripts.create_database import create_database
from scripts.add_station_metadata import add_station_metadata

def main():
    print("Step 1: Creating database...")
    create_database.create_database()
    
    print("Step 2: Adding station metadata into the database...")
    add_station_metadata.main()
    
    print("Step 3: Processing and saving observations as Parquet files...")
    add_observations.main()
    
if __name__ == "__main__":
    main()