import pandas as pd

daily = pd.read_parquet("data/processed/daily.parquet")
stations = pd.read_parquet("data/processed/stations.parquet")

print("\n===== STATE COVERAGE =====")
print(daily.groupby("State").agg(
    records=("Station", "count"),
    stations=("Station", "nunique"),
    districts=("District", "nunique"),
    first_date=("Date", "min"),
    last_date=("Date", "max")
).to_string())

print("\n===== TOTAL COVERAGE =====")
print("Total states:", daily["State"].nunique())
print("Total districts:", daily["District"].nunique())
print("Total stations:", stations["Station"].nunique())

print("\n===== MODEL COMPARISON =====")
print(pd.read_csv(
    "data/processed/model_comparison.csv"
).to_string(index=False))

print("\n===== BACKTEST METRICS =====")
print(pd.read_csv(
    "data/processed/backtest_metrics.csv"
).to_string(index=False))

print("\n===== DEPTH DATA QUALITY =====")
print("Minimum depth:", daily["Depth_m"].min())
print("Maximum depth:", daily["Depth_m"].max())
print("Negative values:", (daily["Depth_m"] < 0).sum())
print("Missing values:", daily["Depth_m"].isna().sum())

print("\n===== DATE CHECK =====")
print("Earliest:", daily["Date"].min())
print("Latest:", daily["Date"].max())

print("\n===== STATE RECORD COUNTS =====")
print(daily.groupby("State").size())

print("\n===== DEPTH DISTRIBUTION BY STATE =====")

print(
    daily.groupby("State")["Depth_m"]
    .describe(percentiles=[0.50, 0.90, 0.95, 0.99, 0.999])
    .to_string()
)

print("\n===== HIGH DEPTH RECORDS (ABOVE 50m) =====")

high_depth = daily[daily["Depth_m"] > 50]

print("Records above 50m:", len(high_depth))

print(
    high_depth[
        ["State", "District", "Station", "Date", "Depth_m"]
    ].head(20).to_string(index=False)
)

print("\n===== HIGHEST 10 DEPTH RECORDS =====")

print(
    daily.nlargest(10, "Depth_m")[
        ["State", "District", "Station", "Date", "Depth_m"]
    ].to_string(index=False)
)

print("\n===== INVESTIGATE HIGH-DEPTH STATIONS =====")

target_stations = [
    "BATHINDA;MAUR;BAGER MOHABAT SINGH;Sit in baghar Mohabat Singh",
    "AMRITSAR;BABA BAKALA ;RAYYA;SATHIALA;Vet. Hospital",
    "BARNALA MAHAL KALAN MAHAL KALAN 200m DEEP Veterinary Hospital"
]

for station_name in target_stations:
    print("\n--- STATION:", station_name, "---")

    station_data = daily[
        daily["Station"] == station_name
    ].sort_values("Date")

    print("Total records:", len(station_data))

    if len(station_data) > 0:
        print(
            station_data[
                ["State", "District", "Station", "Date", "Depth_m"]
            ].tail(15).to_string(index=False)
        )

        print("\nDepth statistics:")
        print(station_data["Depth_m"].describe())

        print("\nMost frequent depth values:")
        print(station_data["Depth_m"].round(3).value_counts().head(10))

        print("\nStation metadata:")
        print(
            stations[stations["Station"] == station_name].to_string(index=False)
        )

print("\n===== STATION QUALITY SUMMARY =====")

print("Total stations:", len(stations))
print("Usable stations:", stations["usable"].sum())
print("Suspect stations:", stations["suspect"].sum())

print("\n===== STATIONS BY STATE =====")

print(
    stations.groupby("State").agg(
        total_stations=("Station", "count"),
        usable=("usable", "sum"),
        suspect=("suspect", "sum")
    ).to_string()
)

print("\n===== RECORDS FROM USABLE STATIONS =====")

usable_ids = stations.loc[
    stations["usable"] == True, "Station"
]

usable_daily = daily[daily["Station"].isin(usable_ids)]

print("Usable station records:", len(usable_daily))
print("Excluded station records:", len(daily) - len(usable_daily))
