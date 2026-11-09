"""P17 Groundwater - step 1: clean the raw telemetry and build daily station series.

Input : data/raw/gwl_tel_*.zip (or .csv)   - Punjab / Bihar telemetry exports
Output: data/processed/daily.parquet       - one row per station per observed day
        data/processed/stations.parquet    - one row per station (metadata + quality flags)
        data/processed/quality_report.json - exactly what was dropped and why

Convention used everywhere in this project
------------------------------------------
Depth_m = depth to water BELOW GROUND in metres (bigger number = deeper = worse).
Bihar files already store positive depths. Punjab files mix conventions between
stations (many are negative depths, some are positive), so the *magnitude* is used.
Readings with magnitude >= 150 m (999 / 958 style placeholders, elevation values)
and exact zeros are treated as sensor errors and removed.
Nothing is interpolated: a day with no valid reading simply has no row.
"""
from pathlib import Path
import io
import json
import zipfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"

MAX_DEPTH_M = 150.0        # anything >= this is a placeholder / elevation value, not a depth
HAMPEL_WINDOW = "15D"      # centred rolling window for the spike filter (daily values)
HAMPEL_MIN_DEV_M = 5.0     # a daily value > 5 m away from its 15-day median is a glitch
MIN_DAYS_USABLE = 60       # stations with fewer observed days are kept but not modelled
MIN_DAYS_PER_YEAR = 20     # a station-year with fewer observed days is treated as stray readings
STUCK_STD_M = 0.05         # a station whose depth never moves (std below this) is a stuck sensor
SUSPECT_STD_M = 15.0       # station-level std above this => unreliable sensor, not modelled
SUSPECT_MEDIAN_M = 90.0    # median depth above this => placeholder-like values, not modelled

USECOLS = ["Station", "State", "District", "Tehsil", "Block", "Village",
           "Latitude", "Longitude", "Data Acquisition Time"]


def _sources():
    files = sorted(list(RAW.glob("gwl_tel_*.zip")) + list(RAW.glob("gwl_tel_*.csv")) + list(RAW.glob("gwl_tel_*.txt")))
    if not files:
        raise FileNotFoundError(f"No gwl_tel_* telemetry files found in {RAW}")
    return files


def _read_one(path: Path) -> pd.DataFrame:
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as zf:
            name = [n for n in zf.namelist() if n.lower().endswith((".csv", ".txt"))][0]
            raw = zf.read(name)
        # the 2021-2025 export is tab-separated; the others are comma-separated
        sep = "\t" if raw[:2000].count(b"\t") > raw[:2000].count(b",") else ","
        df = pd.read_csv(io.BytesIO(raw), sep=sep, low_memory=False)
    else:
        df = pd.read_csv(path, sep=None, engine="python") if path.stat().st_size < 5e6 else pd.read_csv(path, low_memory=False)
    value_col = [c for c in df.columns if c.startswith("Groundwater Level Telemetry")][0]
    df = df[USECOLS + [value_col]].rename(columns={value_col: "raw_value",
                                                  "Data Acquisition Time": "Timestamp"})
    return df


def clean_readings(df: pd.DataFrame, tag: str):
    """Row-level cleaning. Returns cleaned readings + a log dict."""
    log = {"file": tag, "rows_raw": int(len(df))}
    # Two timestamp spellings occur (the 2021-2025 file was passed through Excel): 13-01-2021 06:00 and 1/1/2021 0:00.
    # Both are day-first; this was checked by confirming every station's series is time-ordered under d/m.
    raw_t = df["Timestamp"].astype(str)
    df["Timestamp"] = pd.to_datetime(raw_t, format="%d-%m-%Y %H:%M", errors="coerce").fillna(
        pd.to_datetime(raw_t, format="%d/%m/%Y %H:%M", errors="coerce"))
    df["raw_value"] = pd.to_numeric(df["raw_value"], errors="coerce")
    bad = df["Timestamp"].isna() | df["raw_value"].isna() | ~np.isfinite(df["raw_value"])
    log["dropped_unparseable_time_or_value"] = int(bad.sum())
    df = df[~bad].copy()
    log["negative_values_share"] = round(float((df["raw_value"] < 0).mean()), 4)
    df["Depth_m"] = df["raw_value"].abs()
    big = df["Depth_m"] >= MAX_DEPTH_M
    log["dropped_placeholder_ge_150m"] = int(big.sum())
    df = df[~big]
    zero = df["Depth_m"] == 0
    log["dropped_exact_zero"] = int(zero.sum())
    df = df[~zero]
    log["rows_kept"] = int(len(df))
    return df, log


def to_daily(df: pd.DataFrame) -> pd.DataFrame:
    df["Date"] = df["Timestamp"].dt.floor("D")
    g = df.groupby(["State", "Station", "Date"], sort=False)["Depth_m"]
    daily = g.agg(Depth_m="median", n_obs="size", Depth_min="min", Depth_max="max").reset_index()
    return daily


def hampel(daily: pd.DataFrame):
    """Drop daily values that sit > HAMPEL_MIN_DEV_M away from the centred 15-day median."""
    keep = np.ones(len(daily), dtype=bool)
    daily = daily.sort_values(["State", "Station", "Date"]).reset_index(drop=True)
    for _, idx in daily.groupby(["State", "Station"], sort=False).indices.items():
        s = daily.loc[idx, ["Date", "Depth_m"]].set_index("Date")["Depth_m"]
        med = s.rolling(HAMPEL_WINDOW, center=True, min_periods=5).median()
        dev = (s - med).abs()
        keep[idx] = ~(dev > HAMPEL_MIN_DEV_M).to_numpy()
    removed = int((~keep).sum())
    return daily[keep].reset_index(drop=True), removed


# plausible coordinate boxes (lat_min, lat_max, lon_min, lon_max)
STATE_BOX = {"Punjab": (29.4, 32.6, 73.8, 77.0), "Bihar": (24.2, 27.6, 83.2, 88.4)}


def fix_coordinates(meta: pd.DataFrame) -> pd.DataFrame:
    """A few Punjab rows have corrupt coordinates (e.g. longitude 76163611 = 76.163611 without its decimal point).
    Rescale obvious decimal slips, then replace anything still outside the state's box with the district median."""
    meta = meta.copy()
    for c in ("Latitude", "Longitude"):
        meta[c] = pd.to_numeric(meta[c], errors="coerce")
        big = meta[c].abs() > 1000
        meta.loc[big, c] = meta.loc[big, c] / 1e6
    meta["coord_fixed"] = False
    for state, (la0, la1, lo0, lo1) in STATE_BOX.items():
        m = meta["State"] == state
        ok = meta["Latitude"].between(la0, la1) & meta["Longitude"].between(lo0, lo1)
        bad = m & ~ok
        if bad.any():
            med = meta[m & ok].groupby("District")[["Latitude", "Longitude"]].median()
            for i in meta.index[bad]:
                d = meta.at[i, "District"]
                if d in med.index:
                    meta.at[i, "Latitude"], meta.at[i, "Longitude"] = med.loc[d]
                else:
                    meta.at[i, "Latitude"], meta.at[i, "Longitude"] = meta[m & ok][["Latitude", "Longitude"]].median()
                meta.at[i, "coord_fixed"] = True
    return meta


def station_labels(meta: pd.DataFrame) -> pd.Series:
    def one(r):
        st = str(r["Station"])
        village = str(r["Village"]).strip().title()
        for alt in ("Block", "Tehsil", "District"):
            if village not in {"-", "Nan", "None", ""}:
                break
            village = str(r[alt]).strip().title()
        if ";" in st:                       # Punjab: DISTRICT;TEHSIL;BLOCK;VILLAGE;SITE
            p = [x.strip() for x in st.split(";")]
            base = f"{p[3].title()} - {p[4]}" if len(p) >= 5 else village
        elif st.startswith("PB_"):          # PB_Amritsar_Chogawan_kohali
            p = st.split("_")
            base = " ".join(x.title() for x in p[2:]) or village
        else:                               # Bihar: BHGW-0001
            base = f"{village} ({st})" if len(st) <= 14 else village
        return base
    lab = meta.apply(one, axis=1)
    dup = lab.duplicated(keep=False)
    lab[dup] = lab[dup] + " [" + meta.loc[dup, "Block"].astype(str).str.title() + "]"
    dup = lab.duplicated(keep=False)
    lab[dup] = lab[dup] + " #" + (lab[dup].groupby(lab[dup]).cumcount() + 1).astype(str)
    return lab


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    daily_parts, meta_parts, logs = [], [], []
    for path in _sources():
        print(f"Reading {path.name} ...", flush=True)
        raw = _read_one(path)
        meta_parts.append(raw.drop_duplicates("Station")[["Station", "State", "District", "Tehsil",
                                                           "Block", "Village", "Latitude", "Longitude"]])
        clean, log = clean_readings(raw, path.name)
        del raw
        daily_parts.append(to_daily(clean))
        logs.append(log)
        print(f"   {log['rows_raw']:,} rows -> {log['rows_kept']:,} valid readings", flush=True)

    daily = pd.concat(daily_parts, ignore_index=True)
    # a station can appear in two files (e.g. Punjab 2020 + 2026): median-merge any same-day overlap
    daily = (daily.groupby(["State", "Station", "Date"], as_index=False)
             .agg(Depth_m=("Depth_m", "median"), n_obs=("n_obs", "sum"),
                  Depth_min=("Depth_min", "min"), Depth_max=("Depth_max", "max")))
    n_before = len(daily)
    daily, n_spikes = hampel(daily)
    # isolated stray readings (e.g. 1 reading in 1999 for a station that really reports from 2020)
    # would stretch a station's date range; drop any station-year with fewer than 20 observed days.
    yr_n = daily.groupby(["State", "Station", daily["Date"].dt.year])["Date"].transform("size")
    n_stray = int((yr_n < MIN_DAYS_PER_YEAR).sum())
    daily = daily[yr_n >= MIN_DAYS_PER_YEAR].reset_index(drop=True)
    print(f"Spike filter removed {n_spikes:,} of {n_before:,} station-days", flush=True)

    meta = (pd.concat(meta_parts).drop_duplicates(["State", "Station"]).reset_index(drop=True))
    meta["District"] = meta["District"].astype(str).str.strip().str.title()
    meta["Block"] = meta["Block"].astype(str).str.strip()
    meta["Tehsil"] = meta["Tehsil"].astype(str).str.strip()
    meta["Village"] = meta["Village"].astype(str).str.strip()
    meta = fix_coordinates(meta)
    meta["StationLabel"] = station_labels(meta)

    g = daily.groupby(["State", "Station"])
    q = g.agg(first_date=("Date", "min"), last_date=("Date", "max"), obs_days=("Date", "size"),
              median_depth=("Depth_m", "median"), depth_std=("Depth_m", "std")).reset_index()
    # Coverage is measured inside each calendar year a station reports in, so a station that
    # has 2020 and 2026 data is not penalised for the 2021-2025 gap that is missing from the files.
    yr = daily.assign(Year=daily["Date"].dt.year).groupby(["State", "Station", "Year"])["Date"]
    ys = yr.agg(n="size", lo="min", hi="max").reset_index()
    ys["span"] = (ys["hi"] - ys["lo"]).dt.days + 1
    cov = ys.groupby(["State", "Station"]).agg(n=("n", "sum"), span=("span", "sum")).reset_index()
    cov["coverage"] = (cov["n"] / cov["span"]).round(3)
    q = q.merge(cov[["State", "Station", "coverage"]], on=["State", "Station"])
    q["span_days"] = (q["last_date"] - q["first_date"]).dt.days + 1
    # "suspect" = readings jump around by tens of metres or sit at placeholder-like depths.
    q["suspect"] = ((q["depth_std"] > SUSPECT_STD_M) | (q["median_depth"] >= SUSPECT_MEDIAN_M)
                    | ((q["depth_std"] < STUCK_STD_M) & (q["obs_days"] >= MIN_DAYS_USABLE)))
    q["usable"] = (q["obs_days"] >= MIN_DAYS_USABLE) & ~q["suspect"]
    stations = meta.merge(q, on=["State", "Station"], how="inner")
    stations = stations.sort_values(["State", "District", "StationLabel"]).reset_index(drop=True)

    daily = daily.merge(stations[["State", "Station", "District"]], on=["State", "Station"])
    daily["Date"] = pd.to_datetime(daily["Date"])
    daily = daily[["State", "District", "Station", "Date", "Depth_m", "n_obs", "Depth_min", "Depth_max"]]
    daily = daily.sort_values(["State", "Station", "Date"]).reset_index(drop=True)
    for c in ["State", "District", "Station"]:
        daily[c] = daily[c].astype("category")
    daily.to_parquet(OUT / "daily.parquet", index=False)
    stations.to_parquet(OUT / "stations.parquet", index=False)

    report = {
        "files": logs,
        "spike_filter": {"window": HAMPEL_WINDOW, "threshold_m": HAMPEL_MIN_DEV_M,
                         "station_days_before": int(n_before), "station_days_removed": int(n_spikes),
                         "stray_station_years_removed_station_days": n_stray},
        "final": {
            "station_days": int(len(daily)), "stations": int(len(stations)),
            "stations_usable": int(stations["usable"].sum()),
            "stations_suspect": int(stations["suspect"].sum()),
            "states": sorted(stations["State"].unique().tolist()),
            "districts": int(stations.groupby("State")["District"].nunique().sum()),
            "first_date": str(daily["Date"].min().date()), "last_date": str(daily["Date"].max().date()),
            "by_state": {s: {"stations": int(len(x)), "districts": int(x["District"].nunique()),
                             "station_days": int(x["obs_days"].sum()),
                             "first_date": str(x["first_date"].min().date()),
                             "last_date": str(x["last_date"].max().date())}
                         for s, x in stations.groupby("State")},
        },
        "convention": "Depth_m = depth below ground (m), magnitude of the source value; larger = deeper.",
    }
    (OUT / "quality_report.json").write_text(json.dumps(report, indent=2))
    print("\nDONE")
    print(json.dumps(report["final"], indent=2))


if __name__ == "__main__":
    main()
