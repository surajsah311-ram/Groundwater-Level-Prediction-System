from database_store import _connect

conn, mode = _connect()

try:
    with conn.cursor() as cur:
        for table in [
            "groundwater_stations",
            "groundwater_observations",
            "groundwater_forecasts"
        ]:
            cur.execute(f'SELECT COUNT(*) AS total FROM public."{table}"')
            result = cur.fetchone()

            print(f"{table}: {result['total']} rows")

    print("\nSupabase tables inspected successfully!")

finally:
    conn.close()