"""Local SQLite / optional PostgreSQL store for normalized official groundwater data.

Local development works without credentials using SQLite. For durable cloud storage,
set DATABASE_URL to a private PostgreSQL connection string in deployment secrets.
Never commit the connection string.
"""
from __future__ import annotations

import hashlib
import os
import sqlite3
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # .env loading is optional; environment variables and Streamlit secrets still work.
    pass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parent
DEFAULT_DB = ROOT / "data" / "groundwater_intelligence.sqlite3"
REQUIRED = ["state", "district", "station", "observed_at", "source_value", "measurement_type", "source_file"]


def _setting(name: str, default: str = "") -> str:
    value = os.getenv(name, "").strip()
    if value:
        return value
    try:
        import streamlit as st
        value = str(st.secrets.get(name, "")).strip()
    except Exception:
        value = ""
    return value or default


def _mode() -> str:
    return "postgresql" if _setting("DATABASE_URL") else "sqlite"


def _connect():
    url = _setting("DATABASE_URL")
    if url:
        try:
            import psycopg
            from psycopg.rows import dict_row
        except ImportError as exc:
            raise RuntimeError("PostgreSQL selected but psycopg is missing. Install requirements.txt.") from exc
        conn = psycopg.connect(url, connect_timeout=10, row_factory=dict_row)
        return conn, "postgresql"
    path = Path(_setting("GROUNDWATER_DB_PATH", str(DEFAULT_DB))).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(path, timeout=30), "sqlite"


def _execute(conn, mode: str, sql: str, params=()):
    if mode == "postgresql":
        with conn.cursor() as cur:
            cur.execute(sql.replace("?", "%s"), params)
            return cur
    return conn.execute(sql, params)


def _ensure_schema(conn, mode: str) -> None:
    if mode == "postgresql":
        _execute(conn, mode, """
            CREATE TABLE IF NOT EXISTS groundwater_observations (
              record_id TEXT PRIMARY KEY, state TEXT, district TEXT, block TEXT, tehsil TEXT, village TEXT,
              station TEXT, latitude DOUBLE PRECISION, longitude DOUBLE PRECISION, observed_at TIMESTAMPTZ,
              source_value DOUBLE PRECISION, source_unit TEXT, value_semantics TEXT, measurement_type TEXT,
              source_file TEXT, source_url TEXT, ingested_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """)
        _execute(conn, mode, """
            CREATE TABLE IF NOT EXISTS groundwater_sync_runs (
              id BIGSERIAL PRIMARY KEY, started_at TIMESTAMPTZ NOT NULL, finished_at TIMESTAMPTZ NOT NULL,
              source_file TEXT NOT NULL, rows_written BIGINT NOT NULL, rows_skipped BIGINT NOT NULL, status TEXT NOT NULL
            )
        """)
    else:
        _execute(conn, mode, """
            CREATE TABLE IF NOT EXISTS groundwater_observations (
              record_id TEXT PRIMARY KEY, state TEXT, district TEXT, block TEXT, tehsil TEXT, village TEXT,
              station TEXT, latitude REAL, longitude REAL, observed_at TEXT, source_value REAL,
              source_unit TEXT, value_semantics TEXT, measurement_type TEXT, source_file TEXT, source_url TEXT,
              ingested_at TEXT NOT NULL
            )
        """)
        _execute(conn, mode, """
            CREATE TABLE IF NOT EXISTS groundwater_sync_runs (
              id INTEGER PRIMARY KEY AUTOINCREMENT, started_at TEXT NOT NULL, finished_at TEXT NOT NULL,
              source_file TEXT NOT NULL, rows_written INTEGER NOT NULL, rows_skipped INTEGER NOT NULL, status TEXT NOT NULL
            )
        """)
    conn.commit()


def database_status() -> dict[str, Any]:
    conn, mode = _connect()
    try:
        _ensure_schema(conn, mode)
        row = _execute(conn, mode, "SELECT COUNT(*) AS n FROM groundwater_observations").fetchone()
        if mode == "postgresql":
            last = _execute(conn, mode, "SELECT finished_at FROM groundwater_sync_runs ORDER BY id DESC LIMIT 1").fetchone()
            last_sync = str(last["finished_at"]) if last else ""
        else:
            last = _execute(conn, mode, "SELECT finished_at FROM groundwater_sync_runs ORDER BY id DESC LIMIT 1").fetchone()
            last_sync = str(last[0]) if last else ""
        return {"mode": mode, "configured": True, "observation_rows": int(row["n"] if mode == "postgresql" else row[0]),
                "last_sync_utc": last_sync, "persistent_cloud_storage": mode == "postgresql"}
    finally:
        conn.close()


def _fingerprint(row: dict[str, Any]) -> str:
    # Include source provenance and raw value so repeated imports are idempotent.
    keys = ("state", "district", "station", "observed_at", "source_value", "measurement_type", "source_file")
    raw = "|".join("" if pd.isna(row.get(k)) else str(row.get(k)) for k in keys)
    return hashlib.sha256(raw.encode("utf-8", errors="replace")).hexdigest()


def sync_normalized_csv(csv_path: str | Path, chunksize: int = 10000) -> dict[str, int]:
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"Normalized CSV not found: {path}")
    columns = pd.read_csv(path, nrows=0).columns.tolist()
    missing = [c for c in REQUIRED if c not in columns]
    if missing:
        raise ValueError("Normalized CSV is missing required columns: " + ", ".join(missing))

    conn, mode = _connect()
    started = datetime.now(timezone.utc).isoformat()
    written = skipped = 0
    try:
        _ensure_schema(conn, mode)
        for frame in pd.read_csv(path, chunksize=chunksize, low_memory=False):
            frame["observed_at"] = pd.to_datetime(frame["observed_at"], errors="coerce", utc=True)
            frame["source_value"] = pd.to_numeric(frame["source_value"], errors="coerce")
            valid = frame["observed_at"].notna() & frame["source_value"].notna()
            skipped += int((~valid).sum())
            frame = frame.loc[valid]
            rows = []
            for item in frame.to_dict("records"):
                item["record_id"] = _fingerprint(item)
                for key in ("latitude", "longitude"):
                    try:
                        item[key] = float(item[key]) if pd.notna(item.get(key)) else None
                    except (TypeError, ValueError):
                        item[key] = None
                item["observed_at"] = item["observed_at"].isoformat()
                item["ingested_at"] = datetime.now(timezone.utc).isoformat()
                for key in ("state", "district", "block", "tehsil", "village", "station", "source_unit", "value_semantics", "measurement_type", "source_file", "source_url"):
                    val = item.get(key)
                    item[key] = None if pd.isna(val) or str(val).strip().lower() in {"", "nan", "<na>"} else str(val).strip()
                rows.append(item)
            if not rows:
                continue
            cols = ["record_id", "state", "district", "block", "tehsil", "village", "station", "latitude", "longitude", "observed_at", "source_value", "source_unit", "value_semantics", "measurement_type", "source_file", "source_url", "ingested_at"]
            if mode == "postgresql":
                placeholders = ",".join(["%s"] * len(cols))
                sql = f"INSERT INTO groundwater_observations ({','.join(cols)}) VALUES ({placeholders}) ON CONFLICT (record_id) DO NOTHING"
                with conn.cursor() as cur:
                    cur.executemany(sql, [[r.get(c) for c in cols] for r in rows])
                    written += cur.rowcount if cur.rowcount > 0 else 0
            else:
                placeholders = ",".join(["?"] * len(cols))
                sql = f"INSERT OR IGNORE INTO groundwater_observations ({','.join(cols)}) VALUES ({placeholders})"
                before = conn.total_changes
                conn.executemany(sql, [[r.get(c) for c in cols] for r in rows])
                written += conn.total_changes - before
            conn.commit()
        finished = datetime.now(timezone.utc).isoformat()
        _execute(conn, mode, "INSERT INTO groundwater_sync_runs (started_at, finished_at, source_file, rows_written, rows_skipped, status) VALUES (?, ?, ?, ?, ?, ?)",
                 (started, finished, str(path), written, skipped, "success"))
        conn.commit()
        return {"rows_written": written, "rows_skipped": skipped}
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Sync normalized official groundwater observations into SQLite or PostgreSQL.")
    parser.add_argument("--input", default="data/processed/official/normalized_observations.csv")
    args = parser.parse_args()
    result = sync_normalized_csv(args.input)
    print(f"Database mode: {_mode()}; inserted {result['rows_written']:,}; skipped invalid rows {result['rows_skipped']:,}")
