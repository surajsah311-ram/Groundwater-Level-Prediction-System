import os
import pandas as pd
import psycopg
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
CSV_PATH = "data/processed/official/supabase_sample.csv"

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL missing from .env")

df = pd.read_csv(CSV_PATH)

df["observed_at"] = pd.to_datetime(
    df["observed_at"], errors="coerce", utc=True
)
df["source_value"] = pd.to_numeric(
    df["source_value"], errors="coerce"
)

df = df.dropna(subset=["observed_at", "source_value"])

df["quality_flag"] = "within_screening_range"

columns = [
    "state", "district", "block", "tehsil", "village",
    "station", "latitude", "longitude", "observed_at",
    "source_value", "source_unit", "value_semantics",
    "measurement_type", "source_file", "source_url",
    "quality_flag"
]

df = df[columns]
df = df.astype(object).where(pd.notna(df), None)

insert_sql = """
INSERT INTO official_groundwater_observations (
    state, district, block, tehsil, village,
    station, latitude, longitude, observed_at,
    source_value, source_unit, value_semantics,
    measurement_type, source_file, source_url, quality_flag
)
VALUES (
    %s, %s, %s, %s, %s, %s, %s, %s,
    %s, %s, %s, %s, %s, %s, %s, %s
)
ON CONFLICT (
    source_file, station, observed_at,
    measurement_type, source_value
) DO NOTHING
"""

attempted = 0

with psycopg.connect(DATABASE_URL) as conn:
    with conn.cursor() as cur:
        records = list(df.itertuples(index=False, name=None))

        for start in range(0, len(records), 500):
            batch = records[start:start + 500]
            cur.executemany(insert_sql, batch)
            attempted += len(batch)
            print(f"Processed: {attempted}/{len(records)}")

print("\nSample import finished!")
print("Rows attempted:", attempted)