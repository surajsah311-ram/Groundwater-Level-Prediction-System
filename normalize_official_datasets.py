"""Normalize downloaded official groundwater CSVs into a traceable common schema.

This is an ingestion/inspection step, NOT a promise that all sources are model-ready.
It preserves the original numeric value and measurement type. It does not convert
sign conventions or combine telemetry with quarterly manual readings.

Usage:
  python normalize_official_datasets.py --input data/raw/official --output data/processed/official
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable

import pandas as pd

# Alias matching is deliberately conservative. Source headers remain recorded.
ALIASES = {
    "state": ["state", "state name", "state/ut", "state / ut", "state_name"],
    "district": ["district", "district name", "district_name"],
    "station": ["station", "station name", "station id", "station code", "well name", "monitoring station", "site name", "observation well"],
    "date": ["date", "datetime", "timestamp", "data acquisition time", "observation date", "measurement date", "date time", "sample date"],
    "value": ["groundwater level", "ground water level", "water level below ground", "depth to water", "depth below ground", "groundwater depth", "gwl", "water level (m)", "water level in m"],
    "latitude": ["latitude", "lat"],
    "longitude": ["longitude", "long", "lon"],
    "block": ["block", "block name", "development block"],
    "tehsil": ["tehsil", "taluka", "sub district", "sub-district"],
    "village": ["village", "village name"],
}

STATE_NAMES = [
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh", "Goa", "Gujarat",
    "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka", "Kerala", "Madhya Pradesh",
    "Maharashtra", "Manipur", "Meghalaya", "Mizoram", "Nagaland", "Odisha", "Punjab",
    "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana", "Tripura", "Uttar Pradesh", "Uttarakhand",
    "West Bengal", "Andaman and Nicobar Islands", "Chandigarh", "Dadra and Nagar Haveli and Daman and Diu",
    "Delhi", "Jammu and Kashmir", "Ladakh", "Lakshadweep", "Puducherry",
]


def norm(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).strip().lower()).strip()


def find_column(columns: Iterable[object], key: str) -> object | None:
    cols = list(columns)
    normalized = {norm(c): c for c in cols}
    for alias in ALIASES[key]:
        a = norm(alias)
        if a in normalized:
            return normalized[a]
    # Fuzzy matching is limited to the value field to avoid mistaking RL/altitude for groundwater.
    if key == "value":
        for c in cols:
            n = norm(c)
            if ("groundwater" in n or "ground water" in n or "depth to water" in n or "water level below ground" in n) and not any(x in n for x in ("rain", "river", "rl msl", "elevation", "altitude")):
                return c
    return None


def infer_state(path: Path, manifest_item: dict | None) -> str:
    # Use only explicit state names in a source filename/path or catalogue label.
    text = " ".join([str(path), str((manifest_item or {}).get("label", "")), str((manifest_item or {}).get("url", ""))]).lower()
    for state in sorted(STATE_NAMES, key=len, reverse=True):
        if state.lower() in text or state.lower().replace(" ", "-") in text or state.lower().replace(" ", "_") in text:
            return state
    abbreviations = {"pb": "Punjab", "br": "Bihar", "as": "Assam", "up": "Uttar Pradesh", "mp": "Madhya Pradesh", "wb": "West Bengal", "tn": "Tamil Nadu"}
    tokens = set(re.split(r"[^a-z]+", text))
    for token, state in abbreviations.items():
        if token in tokens:
            return state
    return ""


def measurement_type(path: Path, item: dict | None) -> str:
    kind = str((item or {}).get("kind", "")).lower()
    text = (str(path) + " " + str((item or {}).get("label", ""))).lower()
    if kind == "manual_quarterly" or "quarterly" in text or "manual" in text:
        return "manual_quarterly"
    if kind == "telemetry" or "telemetry" in text or "tel_" in text:
        return "telemetry"
    return "unknown"


def parse_dates(values: pd.Series) -> pd.Series:
    """Parse ISO dates without swapping month/day; fall back to Indian day-first formats."""
    text = values.astype("string").str.strip()
    sample = next((str(v) for v in text.dropna().head(20) if str(v)), "")
    # ISO-like YYYY-MM-DD or YYYY/MM/DD should always be parsed year-first.
    iso_year_first = bool(re.match(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}", sample))
    parsed = pd.to_datetime(text, errors="coerce", dayfirst=not iso_year_first, utc=True)
    if parsed.isna().any():
        fallback = pd.to_datetime(text[parsed.isna()], errors="coerce", dayfirst=True, utc=True)
        parsed.loc[parsed.isna()] = fallback
    return parsed.dt.tz_convert(None)


def unit_from_header(header: object) -> str:
    n = norm(header)
    if "millimeter" in n or " mm " in f" {n} ":
        return "mm"
    if "meter" in n or "metre" in n or re.search(r"\(\s*m\s*\)", str(header), re.I):
        return "m"
    return "unknown"


def load_manifest(root: Path) -> dict[str, dict]:
    path = root / "download_manifest.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    out = {}
    for item in data if isinstance(data, list) else []:
        local = str(item.get("local_path", "")).replace("\\", "/")
        if local:
            out[local] = item
            out[Path(local).name] = item
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default="data/raw/official", help="Folder containing downloaded source CSVs")
    ap.add_argument("--output", default="data/processed/official", help="Folder for normalized data and audit reports")
    ap.add_argument("--chunksize", type=int, default=100_000)
    args = ap.parse_args()
    source_root = Path(args.input).resolve()
    out_root = Path(args.output).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(source_root)
    csvs = [p for p in sorted(source_root.rglob("*.csv")) if not p.name.lower().startswith("download_manifest") and out_root not in p.parents]
    if not csvs:
        print(f"No source CSV files found in {source_root}. Download official resources first.")
        return 2

    combined = out_root / "normalized_observations.csv"
    combined.unlink(missing_ok=True)
    audits: list[dict] = []
    combined_header_written = False
    for idx, path in enumerate(csvs, 1):
        rel = path.relative_to(source_root).as_posix()
        item = manifest.get(rel) or manifest.get(path.name) or {}
        audit = {"source_file": rel, "source_url": item.get("url", ""), "measurement_type": measurement_type(path, item),
                 "bytes": path.stat().st_size, "status": "pending", "rows_seen": 0, "rows_with_date": 0,
                 "rows_with_value": 0, "first_date": "", "last_date": "", "state_values": "", "district_count": 0,
                 "station_count": 0, "value_column": "", "date_column": "", "notes": ""}
        try:
            header = pd.read_csv(path, nrows=0, encoding_errors="replace")
            cols = list(header.columns)
            mapped = {key: find_column(cols, key) for key in ALIASES}
            if mapped["date"] is None or mapped["value"] is None:
                audit["status"] = "needs_manual_mapping"
                audit["notes"] = "Could not confidently identify both date and groundwater value columns; source not normalized."
                audit["columns"] = " | ".join(map(str, cols))
                audits.append(audit)
                print(f"[{idx}/{len(csvs)}] REVIEW schema: {path.name}")
                continue
            audit["date_column"] = str(mapped["date"])
            audit["value_column"] = str(mapped["value"])
            inferred_state = infer_state(path, item)
            date_min, date_max = None, None
            state_values: set[str] = set()
            districts: set[str] = set()
            stations: set[str] = set()
            counts = {"seen": 0, "date": 0, "value": 0}
            source_value_name = str(mapped["value"])
            unit = unit_from_header(source_value_name)
            semantic = "depth_below_ground" if any(x in norm(source_value_name) for x in ("depth to water", "depth below ground", "water level below ground", "groundwater depth")) else "source_level_semantics_unverified"
            for chunk in pd.read_csv(path, chunksize=max(1, args.chunksize), low_memory=False, encoding_errors="replace"):
                counts["seen"] += len(chunk)
                date = parse_dates(chunk[mapped["date"]])
                value = pd.to_numeric(chunk[mapped["value"]].astype(str).str.replace(",", "", regex=False), errors="coerce")
                state_series = chunk[mapped["state"]].astype("string").str.strip() if mapped["state"] is not None else pd.Series([inferred_state] * len(chunk), index=chunk.index, dtype="string")
                district_series = chunk[mapped["district"]].astype("string").str.strip() if mapped["district"] is not None else pd.Series([pd.NA] * len(chunk), index=chunk.index, dtype="string")
                station_series = chunk[mapped["station"]].astype("string").str.strip() if mapped["station"] is not None else pd.Series([pd.NA] * len(chunk), index=chunk.index, dtype="string")
                valid_dates = date.dropna()
                if not valid_dates.empty:
                    lo, hi = valid_dates.min(), valid_dates.max()
                    date_min = lo if date_min is None or lo < date_min else date_min
                    date_max = hi if date_max is None or hi > date_max else date_max
                counts["date"] += int(date.notna().sum())
                counts["value"] += int(value.notna().sum())
                state_values.update(str(x) for x in state_series.dropna().unique() if str(x).strip())
                districts.update(str(x) for x in district_series.dropna().unique() if str(x).strip())
                stations.update(str(x) for x in station_series.dropna().unique() if str(x).strip())
                normalized = pd.DataFrame({
                    "state": state_series,
                    "district": district_series,
                    "block": chunk[mapped["block"]].astype("string").str.strip() if mapped["block"] is not None else pd.NA,
                    "tehsil": chunk[mapped["tehsil"]].astype("string").str.strip() if mapped["tehsil"] is not None else pd.NA,
                    "village": chunk[mapped["village"]].astype("string").str.strip() if mapped["village"] is not None else pd.NA,
                    "station": station_series,
                    "latitude": pd.to_numeric(chunk[mapped["latitude"]], errors="coerce") if mapped["latitude"] is not None else pd.NA,
                    "longitude": pd.to_numeric(chunk[mapped["longitude"]], errors="coerce") if mapped["longitude"] is not None else pd.NA,
                    "observed_at": date,
                    "source_value": value,
                    "source_unit": unit,
                    "value_semantics": semantic,
                    "measurement_type": audit["measurement_type"],
                    "source_file": rel,
                    "source_url": audit["source_url"],
                    "source_date_column": str(mapped["date"]),
                    "source_value_column": source_value_name,
                })
                normalized.to_csv(combined, mode="a", index=False, header=not combined_header_written, date_format="%Y-%m-%dT%H:%M:%S")
                combined_header_written = True
            audit.update({"status": "normalized", "rows_seen": counts["seen"], "rows_with_date": counts["date"],
                          "rows_with_value": counts["value"], "first_date": date_min.date().isoformat() if date_min is not None else "",
                          "last_date": date_max.date().isoformat() if date_max is not None else "", "state_values": " | ".join(sorted(state_values)),
                          "district_count": len(districts), "station_count": len(stations), "source_unit": unit,
                          "value_semantics": semantic, "columns": " | ".join(map(str, cols)),
                          "notes": "Source numeric values preserved. Verify units/sign convention and semantics before modeling; no missing readings were imputed."})
            print(f"[{idx}/{len(csvs)}] normalized {path.name}: {counts['seen']:,} rows")
        except Exception as exc:
            audit["status"] = "error"
            audit["notes"] = f"{type(exc).__name__}: {exc}"
            print(f"[{idx}/{len(csvs)}] ERROR {path.name}: {exc}")
        audits.append(audit)

    audit_path = out_root / "normalization_audit.csv"
    pd.DataFrame(audits).to_csv(audit_path, index=False)
    summary = {"input_root": str(source_root), "source_files_found": len(csvs), "normalized_files": sum(a["status"] == "normalized" for a in audits),
               "needs_manual_mapping": sum(a["status"] == "needs_manual_mapping" for a in audits),
               "errors": sum(a["status"] == "error" for a in audits),
               "combined_file": str(combined) if combined.exists() else None,
               "audit_file": str(audit_path),
               "warning": "Coverage and values are only as complete as the source files. Do not interpret this inventory as proof of all-India coverage."}
    (out_root / "normalization_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 1 if summary["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
