import unittest
import p17lib as L


class RelativeGroundwaterStatusTests(unittest.TestCase):
    def test_higher_watch(self):
        self.assertEqual(L.relative_groundwater_status(0.9, 0.2), "Higher watch")

    def test_monitor_for_deep_or_declining(self):
        self.assertEqual(L.relative_groundwater_status(0.9, 0.0), "Monitor")
        self.assertEqual(L.relative_groundwater_status(0.4, 0.2), "Monitor")

    def test_lower_relative_watch(self):
        self.assertEqual(L.relative_groundwater_status(0.4, -0.1), "Lower relative watch")


if __name__ == "__main__":
    unittest.main()
