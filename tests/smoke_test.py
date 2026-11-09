"""Smoke test: run the dashboard headlessly and fail on any Streamlit exception.

    python tests/smoke_test.py
"""
import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def run(label, setup=None):
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180)
    at.run()
    if setup:
        setup(at)
        at.run()
    errs = [e.value for e in at.exception]
    status = "OK " if not errs else "FAIL"
    print(f"[{status}] {label}")
    for e in errs:
        print("      ", e[:300])
    return not errs


def pick(at, key, value):
    for s in at.selectbox:
        if s.key == key:
            s.set_value(value)
            return
    raise KeyError(key)


ok = True
ok &= run("default load (all tabs render)")
ok &= run("Bihar station + forecast", lambda at: (pick(at, "se_state", "Bihar")))
ok &= run("Punjab, another district", lambda at: pick(at, "se_district", "Ludhiana"))
sys.exit(0 if ok else 1)
