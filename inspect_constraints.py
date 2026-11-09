
import os
import psycopg
from dotenv import load_dotenv

load_dotenv()
url = os.getenv("DATABASE_URL")

if not url:
    raise ValueError("DATABASE_URL missing from .env")

query = """
SELECT
    table_name,
    column_name,
    is_identity,
    identity_generation,
    column_default
FROM information_schema.columns
WHERE table_schema = 'public'
  AND column_name = 'id'
  AND table_name IN (
      'groundwater_stations',
      'groundwater_observations',
      'groundwater_forecasts'
  )
ORDER BY table_name;
"""

with psycopg.connect(url) as conn:
    with conn.cursor() as cur:
        cur.execute(query)
        print("===== ID GENERATION DETAILS =====")
        for row in cur.fetchall():
            print(row)
