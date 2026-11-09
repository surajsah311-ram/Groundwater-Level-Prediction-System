"""Discover and download official CGWB groundwater CSV resources from NWIC/NWDP.

Downloads the official resource links listed on the telemetry and quarterly-manual
catalogue pages. Large downloads are streamed and a manifest is written. This does
not automatically make all resources model-ready; schemas must be validated first.

Usage:
  python download_official_datasets.py --output data/raw/official
  python download_official_datasets.py --output data/raw/official --include-current
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

CATALOGUES = {
    "telemetry": "https://nwdp.nwic.gov.in/en/dataset/ground-water-level-telemetry-daily-cgwb-as-assam",
    "manual_quarterly": "https://nwdp.nwic.gov.in/dataset/gwl-manual-quarterly-central-ground-water-board-department",
}
USER_AGENT = "P17-Groundwater-Intelligence/1.0 (educational research; respectful download rate)"


def safe_name(url: str, fallback: str) -> str:
    name = Path(urlparse(url).path).name
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    return name if name.lower().endswith(".csv") else fallback + ".csv"


def discover(session: requests.Session, kind: str, page_url: str) -> list[dict]:
    response = session.get(page_url, timeout=45)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    found = {}
    for a in soup.find_all("a", href=True):
        href = urljoin(page_url, a["href"])
        path = urlparse(href).path.lower()
        label = " ".join(a.get_text(" ", strip=True).split())
        if "/download/" not in path or not path.endswith(".csv"):
            continue
        if "nwdp.nwic.gov.in" not in urlparse(href).netloc:
            continue
        if "2026" in label or "2026" in path:
            period = "2026-plus"
        elif "2021" in label or "2021" in path:
            period = "2021-2025"
        elif "1991" in label or "1991" in path:
            period = "1991-2020"
        else:
            period = "unspecified"
        found[href] = {"kind": kind, "catalogue_url": page_url, "label": label or Path(path).name,
                       "url": href, "period_group": period}
    return list(found.values())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/raw/official", help="Output folder for official CSV resources")
    parser.add_argument("--include-current", action="store_true", help="Also download resources labelled 2026 onwards")
    parser.add_argument("--pause", type=float, default=0.25, help="Pause between downloads in seconds")
    args = parser.parse_args()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    resources: list[dict] = []
    for kind, page in CATALOGUES.items():
        try:
            items = discover(session, kind, page)
            print(f"Discovered {len(items)} CSV resources in {kind} catalogue")
            resources.extend(items)
        except requests.RequestException as exc:
            print(f"ERROR: could not read catalogue {page}: {exc}")

    if not resources:
        print("No resources found. Check network access or the official catalogue page structure.")
        return 2

    manifest = []
    for i, item in enumerate(resources, start=1):
        if item["period_group"] == "2026-plus" and not args.include_current:
            item["status"] = "skipped_current_period"
            manifest.append(item)
            continue
        subdir = out / item["kind"] / item["period_group"]
        subdir.mkdir(parents=True, exist_ok=True)
        name = safe_name(item["url"], f"resource_{i:04d}")
        dest = subdir / name
        item["local_path"] = str(dest.relative_to(out))
        if dest.exists() and dest.stat().st_size > 0:
            item["status"] = "already_downloaded"
            item["bytes"] = dest.stat().st_size
            print(f"[{i}/{len(resources)}] exists {dest.name}")
            manifest.append(item)
            continue
        try:
            with session.get(item["url"], stream=True, timeout=(20, 180)) as r:
                r.raise_for_status()
                ctype = r.headers.get("Content-Type", "")
                if "text/html" in ctype.lower():
                    raise ValueError("server returned HTML instead of CSV")
                tmp = dest.with_suffix(dest.suffix + ".part")
                size = 0
                with tmp.open("wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
                            size += len(chunk)
                if size == 0:
                    tmp.unlink(missing_ok=True)
                    raise ValueError("empty download")
                tmp.replace(dest)
            item["status"] = "downloaded"
            item["bytes"] = size
            print(f"[{i}/{len(resources)}] downloaded {dest.name} ({size:,} bytes)")
        except (requests.RequestException, OSError, ValueError) as exc:
            item["status"] = "failed"
            item["error"] = str(exc)
            print(f"[{i}/{len(resources)}] FAILED {item['label']}: {exc}")
        manifest.append(item)
        time.sleep(max(0.0, args.pause))

    manifest_path = out / "download_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    summary_path = out / "download_manifest.csv"
    fields = ["kind", "label", "period_group", "status", "bytes", "local_path", "url", "error"]
    with summary_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(manifest)
    ok = sum(x.get("status") in {"downloaded", "already_downloaded"} for x in manifest)
    failed = sum(x.get("status") == "failed" for x in manifest)
    skipped = sum(x.get("status") == "skipped_current_period" for x in manifest)
    print(f"Done. available={ok}, failed={failed}, skipped-current={skipped}")
    print(f"Manifest: {manifest_path}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
