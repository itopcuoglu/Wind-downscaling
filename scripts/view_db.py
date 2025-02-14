import sqlite3
from config import DB_NAME

def list_tables():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    conn.close()
    return tables

if __name__ == "__main__":
    tables = list_tables()
    if tables:
        print("Tables in the database:")
        for table in tables:
            print(f" - {table[0]}")
    else:
        print("No tables found in the database.")