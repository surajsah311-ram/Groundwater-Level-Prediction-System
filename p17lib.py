"""Shared helpers: rainfall normals, feature construction, model loading.

The SAME `segment_features` function is used for training and for the dashboard forecast,
so what the model saw during training is exactly what it is given at prediction time.
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
MODELS = ROOT / "models"

HORIZONS = [7, 14, 30, 60, 90]
GAP_BREAK_DAYS = 60          # a gap longer than this splits a station's history into segments

# telemetry district name  ->  name used in district_wise_rainfall_normal.csv
# (Fazilka was cut from Ferozepur and Pathankot from Gurdaspur after the rainfall normals were made.)
RAIN_NAME = {
    "Bettiah": "WEST CHAMPARAN", "Chapra": "SARAN", "Jehanabad": "JAHANABAD", "Kaimur": "BHABUA",
    "Motihari": "EAST CHAMPARAN", "Purnia": "PURNEA",
    "Fatehgarh Sahib": "FATEHGARH SAH", "Fazilka": "FEROZEPUR", "Mohali": "SAS NAGAR(MGA)",
    "Mukatsar": "MUKTSAR", "Nawan Shahr": "NAWANSHAHR", "Pathankot": "GURDASPUR", "Ropar": "RUPNAGAR",
}
MONTHS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
_DAYS = np.array([31, 28.25, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])

BASE_FEATURES = ["cur", "d7", "d14", "d30", "d60", "std14", "doy_sin", "doy_cos",
                 "lat", "lon", "rain_ann", "rain_jjas", "rain_prev30"]
FEATURES = BASE_FEATURES + ["rain_next_h"]
FEATURE_LABELS = {
    "cur": "Current level (7-day mean depth)", "d7": "Change over last 7 days",
    "d14": "Change over last 14 days", "d30": "Change over last 30 days",
    "d60": "Change over last 60 days", "std14": "Recent variability (14-day std)",
    "doy_sin": "Season (sin of day-of-year)", "doy_cos": "Season (cos of day-of-year)",
    "lat": "Latitude", "lon": "Longitude", "rain_ann": "District annual normal rainfall",
    "rain_jjas": "District monsoon (Jun-Sep) normal rainfall",
    "rain_prev30": "Normal rainfall, previous 30 days", "rain_next_h": "Normal rainfall, next h days",
}


# ----------------------------------------------------------------------------- rainfall
def load_rain_normals() -> pd.DataFrame:
    r = pd.read_csv(RAW / "district_wise_rainfall_normal.csv")
    r = r[r["STATE_UT_NAME"].isin(["PUNJAB", "BIHAR"])].copy()
    r["key"] = r["DISTRICT"].str.upper().str.strip()
    return r.set_index("key")


def rain_key(district: str) -> str:
    return RAIN_NAME.get(district, district).upper()


def daily_rain_curve(row: pd.Series) -> np.ndarray:
    """366-day normal rainfall curve (mm/day) built by spreading each monthly normal evenly."""
    per_day = np.array([row[m] for m in MONTHS], dtype=float) / _DAYS
    month_of_doy = np.repeat(np.arange(12), np.round(_DAYS).astype(int))[:366]
    month_of_doy = np.pad(month_of_doy, (0, 366 - len(month_of_doy)), constant_values=11)
    return per_day[month_of_doy]


def rain_static(district: str, normals: pd.DataFrame):
    """-> (annual normal mm, Jun-Sep normal mm, cumulative curve over 3 tiled years)."""
    row = normals.loc[rain_key(district)]
    curve = daily_rain_curve(row)
    cum = np.concatenate([[0.0], np.cumsum(np.tile(curve, 3))])
    return float(row["ANNUAL"]), float(row["Jun-Sep"]), cum


def rain_window(cum, doy, start_off, end_off):
    """Normal rainfall (mm) between day-of-year+start_off and +end_off (vectorised)."""
    base = 366 + doy.astype(int)
    return cum[base + end_off] - cum[base + start_off]


# ----------------------------------------------------------------------------- features
def split_segments(dates: np.ndarray):
    """Indices that split a sorted date array wherever the gap exceeds GAP_BREAK_DAYS."""
    gaps = np.diff(dates.astype("datetime64[D]")).astype(int)
    cuts = np.where(gaps > GAP_BREAK_DAYS)[0] + 1
    return np.split(np.arange(len(dates)), cuts)


def segment_features(dates: np.ndarray, depth: np.ndarray, lat: float, lon: float,
                     rain_ann: float, rain_jjas: float, cum: np.ndarray,
                     horizons=HORIZONS, with_targets=True) -> pd.DataFrame:
    """Feature table for one station segment on a daily grid (missing days stay NaN)."""
    idx = pd.date_range(dates[0], dates[-1], freq="D")
    s = pd.Series(depth, index=pd.DatetimeIndex(dates)).reindex(idx)
    S = s.rolling(7, min_periods=3).mean()
    f = pd.DataFrame(index=idx)
    f["cur"] = S
    for k in (7, 14, 30, 60):
        f[f"d{k}"] = S - S.shift(k)
    f["std14"] = s.rolling(14, min_periods=5).std()
    doy = idx.dayofyear.to_numpy()
    f["doy_sin"] = np.sin(2 * np.pi * doy / 366.0)
    f["doy_cos"] = np.cos(2 * np.pi * doy / 366.0)
    f["lat"], f["lon"] = lat, lon
    f["rain_ann"], f["rain_jjas"] = rain_ann, rain_jjas
    f["rain_prev30"] = rain_window(cum, doy, -30, 0)
    for h in horizons:
        f[f"rain_next_{h}"] = rain_window(cum, doy, 0, h)
        if with_targets:
            f[f"y_{h}"] = s.shift(-h) - S
            f[f"t_{h}"] = s.shift(-h)
    f = f[f["cur"].notna()]
    f.index.name = "Date"
    return f.reset_index()


def design_matrix(f: pd.DataFrame, h: int) -> pd.DataFrame:
    X = f[BASE_FEATURES].copy()
    X["rain_next_h"] = f[f"rain_next_{h}"].to_numpy()
    return X[FEATURES]


# ----------------------------------------------------------------------------- model io
def load_models():
    meta_path = MODELS / "meta.json"
    if not meta_path.exists():
        return None, {}
    import joblib
    meta = json.loads(meta_path.read_text())
    models = {}
    for state in meta["states"]:
        for h in HORIZONS:
            p = MODELS / f"{state.lower()}_h{h}.joblib"
            if p.exists():
                models[(state, h)] = joblib.load(p)
    return meta, models


def forecast_station(models, meta, state: str, district: str, dates, depth, lat, lon):
    """Direct multi-horizon forecast from the last valid origin of one station.

    Returns (origin_date, origin_level, DataFrame[h, Date, Forecast_m, Low_m, High_m]) or None.
    """
    normals = load_rain_normals()
    ann, jjas, cum = rain_static(district, normals)
    segs = split_segments(np.asarray(dates))
    last = segs[-1]
    f = segment_features(np.asarray(dates)[last], np.asarray(depth)[last], lat, lon, ann, jjas, cum,
                         with_targets=False)
    if f.empty:
        return None
    row = f.iloc[[-1]]
    origin = pd.Timestamp(row["Date"].iloc[0])
    cur = float(row["cur"].iloc[0])
    out = []
    for h in HORIZONS:
        m = models.get((state, h))
        if m is None:
            continue
        delta = float(m.predict(design_matrix(row, h))[0])
        q = meta["interval"][state][str(h)]
        out.append({"Horizon_days": h, "Date": origin + pd.Timedelta(days=h),
                    "Forecast_m": cur + delta, "Low_m": cur + delta + q["q10"], "High_m": cur + delta + q["q90"]})
    return origin, cur, pd.DataFrame(out)



def relative_groundwater_status(depth_percentile, trend_m_per_year):
    """Within-state screening label; not an official regulatory risk category."""
    import pandas as pd

    deep = pd.notna(depth_percentile) and float(depth_percentile) >= 0.75
    declining = pd.notna(trend_m_per_year) and float(trend_m_per_year) > 0.10
    if deep and declining:
        return "Higher watch"
    if deep or declining:
        return "Monitor"
    return "Lower relative watch"
