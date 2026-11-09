"""Inventory downloaded NWIC/NWDP CSV files without merging incompatible schemas.

Usage: python audit_downloaded_sources.py --input data/raw/official --output data/processed/official_source_inventory.csv
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
import pandas as pd

DATE_CANDIDATES = ["date", "datetime", "timestamp", "data acquisition time", "observation date", "measurement date"]
STATE_CANDIDATES = ["state", "state name", "state/ut"]
DISTRICT_CANDIDATES = ["district", "district name"]
STATION_CANDIDATES = ["station", "station name", "station id", "well name", "monitoring station"]


def find_col(columns, candidates):
    normalized = {str(c).strip().lower(): c for c in columns}
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]
    for c in columns:
        low = str(c).strip().lower()
        if any(candidate in low for candidate in candidates):
            return c
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input', default='data/raw/official')
    ap.add_argument('--output', default='data/processed/official_source_inventory.csv')
    args = ap.parse_args()
    root = Path(args.input)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(root.rglob('*.csv')):
        if path.name.startswith('download_manifest'):
            continue
        row = {'file': str(path), 'bytes': path.stat().st_size, 'status': 'ok'}
        try:
            # Read a small sample first, then date range using only a likely date column.
            sample = pd.read_csv(path, nrows=100, low_memory=False, encoding_errors='replace')
            row['columns'] = len(sample.columns)
            row['column_names'] = ' | '.join(map(str, sample.columns))
            state_col = find_col(sample.columns, STATE_CANDIDATES)
            district_col = find_col(sample.columns, DISTRICT_CANDIDATES)
            station_col = find_col(sample.columns, STATION_CANDIDATES)
            date_col = find_col(sample.columns, DATE_CANDIDATES)
            row['state_column'] = str(state_col or '')
            row['district_column'] = str(district_col or '')
            row['station_column'] = str(station_col or '')
            row['date_column'] = str(date_col or '')
            row['rows'] = sum(1 for _ in path.open('rb')) - 1 if path.stat().st_size < 2_000_000 else ''
            if date_col:
                dates = pd.to_datetime(pd.read_csv(path, usecols=[date_col], low_memory=False,
                                                    encoding_errors='replace')[date_col], errors='coerce', dayfirst=True)
                row['first_date'] = str(dates.min().date()) if dates.notna().any() else ''
                row['last_date'] = str(dates.max().date()) if dates.notna().any() else ''
            else:
                row['first_date'] = row['last_date'] = ''
            row['schema_note'] = 'Review source-specific columns, units and station identifiers before merge.'
        except Exception as exc:
            row['status'] = 'error'
            row['error'] = str(exc)
        rows.append(row)
    pd.DataFrame(rows).to_csv(out, index=False, quoting=csv.QUOTE_MINIMAL)
    print(f'Inventoried {len(rows)} CSV files -> {out}')
    if not rows:
        print('No CSV files found. Run download_official_datasets.py first.')

if __name__ == '__main__':
    main()
