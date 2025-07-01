import sqlite3
from config import DB_NAME

def create_database():
    """
    Creates an SQLite database with a stations table with the following columns:
      - station_id (TEXT PRIMARY KEY)
      - station_name (TEXT)
      - lat (REAL)
      - lon (REAL)
      - height (REAL)
      - elev (REAL)
      - begints (TEXT)
      - endts (TEXT)
      - source_network (TEXT)
      - state (TEXT)
    """
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS stations (
            station_id TEXT PRIMARY KEY,
            station_name TEXT,
            lat REAL,
            lon REAL,
            height REAL,
            elev REAL,
            begints TEXT,
            endts TEXT,
            source_network TEXT,
            state TEXT
        );
    ''')

    conn.commit()
    conn.close()
    print(f"Database '{DB_NAME}' created with table 'stations'.")

if __name__ == "__main__":
    create_database()