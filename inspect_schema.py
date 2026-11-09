
import os
import psycopg
from dotenv import load_dotenv

load_dotenv()

url = os.getenv("DATABASE_URL")

if not url:
    raise ValueError("DATABASE_URL missing from .env")

with psycopg.connect(url) as conn:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT table_name, column_name, data_type
            FROM information_schema.columns
            WHERE table_schema = 'public'
              AND table_name IN (
                  'groundwater_stations',
                  'groundwater_observations',
                  'groundwater_forecasts'
              )
            ORDER BY table_name, ordinal_position;
        """)

        for table, column, data_type in cur.fetchall():
            print(f"{table:30} {column:30} {data_type}")
