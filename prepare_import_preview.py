
import pandas as pd
from pathlib import Path

daily = pd.read_parquet("data/processed/daily.parquet")
stations = pd.read_parquet("data/processed/stations.parquet")

# Keep only observations from usable stations
usable_stations = stations[stations["usable"] == True].copy()

clean_daily = daily[
    daily["Station"].isin(usable_stations["Station"])
].copy()

# Prepare station import preview
station_preview = usable_stations.rename(columns={
    "Station": "source_station_id",
    "StationLabel": "station_name"
})

# Prepare observation import preview
observation_preview = clean_daily.rename(columns={
    "Date": "observed_at",
    "Depth_m": "groundwater_depth_m"
})

output = Path("data/import_preview")
output.mkdir(parents=True, exist_ok=True)

station_preview.to_csv(
    output / "stations_preview.csv", index=False
)

observation_preview.to_csv(
    output / "observations_preview.csv", index=False
)

print("===== IMPORT PREVIEW =====")
print("Stations:", len(station_preview))
print("Observations:", len(observation_preview))
print("States:", clean_daily["State"].nunique())
print("Output folder:", output.resolve())
print("Supabase modified: NO")
