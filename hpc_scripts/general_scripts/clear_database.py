import sqlite3


DB_PATH = "/kfs2/projects/sfcwinds/data/weather_data.db"


with sqlite3.connect(DB_PATH) as conn:
    cursor = conn.cursor()


    cursor.execute("""
        DELETE FROM stations
        WHERE source_network NOT IN ('AZMet', 'CoAgMet');
    """)
    conn.commit()
    print("Cleared stations not in AZMet or CoAgMet.")



