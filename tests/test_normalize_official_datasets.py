"""Unit tests for conservative source normalization using synthetic fixtures only."""
import importlib.util
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "normalize_official_datasets.py"
spec = importlib.util.spec_from_file_location("normalize_official_datasets", SCRIPT)
mod = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(mod)


class NormalizerTests(unittest.TestCase):
    def test_groundwater_value_column_not_rl_msl(self):
        cols = ["Station", "Data Acquisition Time", "RL_MSL", "Groundwater Level Telemetry 6 Hourly (meter)"]
        self.assertEqual(mod.find_column(cols, "value"), cols[3])

    def test_does_not_treat_rainfall_or_elevation_as_groundwater(self):
        self.assertIsNone(mod.find_column(["Date", "Rainfall (mm)", "RL_MSL"], "value"))

    def test_iso_date_is_not_swapped(self):
        parsed = mod.parse_dates(__import__("pandas").Series(["2024-01-02 00:00:00"]))
        self.assertEqual(str(parsed.iloc[0].date()), "2024-01-02")

    def test_day_first_date_is_supported(self):
        parsed = mod.parse_dates(__import__("pandas").Series(["31/12/2024"]))
        self.assertEqual(str(parsed.iloc[0].date()), "2024-12-31")

    def test_measurement_type_from_filename(self):
        self.assertEqual(mod.measurement_type(Path("state_quarterly_manual.csv"), None), "manual_quarterly")
        self.assertEqual(mod.measurement_type(Path("gwl_telemetry.csv"), None), "telemetry")

    def test_state_inference_is_explicit(self):
        self.assertEqual(mod.infer_state(Path("Punjab/gwl_tel_pb.csv"), None), "Punjab")
        self.assertEqual(mod.infer_state(Path("mystery_file.csv"), None), "")


if __name__ == "__main__":
    unittest.main(verbosity=2)
