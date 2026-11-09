import csv
import os
import tempfile
import unittest
from pathlib import Path

import database_store as db


class DatabaseStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmp.name) / "test.sqlite3"
        os.environ["GROUNDWATER_DB_PATH"] = str(self.db_path)
        os.environ.pop("DATABASE_URL", None)
        self.csv_path = Path(self.tmp.name) / "normalized.csv"
        fields = ["state", "district", "block", "tehsil", "village", "station", "latitude", "longitude", "observed_at", "source_value", "source_unit", "value_semantics", "measurement_type", "source_file", "source_url", "source_date_column", "source_value_column"]
        rows = [
            {"state":"Punjab", "district":"Amritsar", "station":"S1", "observed_at":"2024-01-01T00:00:00", "source_value":"12.3", "measurement_type":"telemetry", "source_file":"a.csv", "source_url":"https://example.org/a.csv", "source_unit":"m", "value_semantics":"depth", "latitude":"31.6", "longitude":"74.8"},
            {"state":"Punjab", "district":"Amritsar", "station":"S1", "observed_at":"not-a-date", "source_value":"12.3", "measurement_type":"telemetry", "source_file":"a.csv", "source_url":"https://example.org/a.csv"},
        ]
        with self.csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)

    def tearDown(self):
        self.tmp.cleanup()
        os.environ.pop("GROUNDWATER_DB_PATH", None)

    def test_sync_idempotent_and_skips_invalid(self):
        first = db.sync_normalized_csv(self.csv_path)
        second = db.sync_normalized_csv(self.csv_path)
        status = db.database_status()
        self.assertEqual(first["rows_written"], 1)
        self.assertEqual(first["rows_skipped"], 1)
        self.assertEqual(second["rows_written"], 0)
        self.assertEqual(status["observation_rows"], 1)
        self.assertTrue(status["last_sync_utc"])


if __name__ == "__main__":
    unittest.main()
