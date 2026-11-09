# P17 — Groundwater Intelligence (SIH1696)

Streamlit dashboard + reproducible pipeline for the supplied **Punjab and Bihar** groundwater telemetry:
cleaning, daily station series, district analysis, seasonal/rainfall feature analysis, and **back-tested 7–90 day forecasts**.

## Raw data files

`data/raw/` must contain these four telemetry zips (they are large, so they may be missing from a lightweight copy; copy the originals back before rebuilding) plus the two rainfall CSVs already included:

- `gwl_tel_6_hourly_bihar-gw_br_1991_2020.zip`
- `gwl_tel_6_hourly_punjab-gw_pb_1991_2020.zip`
- `gwl_tel_6_hourly_punjab_2021_2025_partial.zip` (the supplied `21-25.zip`, renamed)
- `gwl_tel_6_hourly_punjab_gw_pb_2026_2030.zip`

Source: India-WRIS telemetric groundwater level data, https://indiawris.gov.in . The dashboard itself runs without them, because the processed data is included.

## Quick start (Windows PowerShell)

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Processed data and trained models are **already included** (`data/processed`, `models`), so the dashboard runs immediately at http://localhost:8501.

To rebuild everything from the raw files in `data/raw` (about 5 minutes):

```powershell
.\.venv\Scripts\python.exe prepare_data.py     # clean raw telemetry -> daily station series
.\.venv\Scripts\python.exe train_models.py     # back-test + train forecast models
.\.venv\Scripts\python.exe build_analysis.py   # district / seasonal / trend tables
.\.venv\Scripts\python.exe compare_models.py   # optional: Ridge / Random Forest comparison (about 10 minutes)
.\.venv\Scripts\python.exe tests\smoke_test.py # headless check that every tab renders
```

## What is in the dashboard

| Tab | Content |
|---|---|
| Overview | Counts, station map coloured by latest depth, coverage per year |
| Station explorer & forecast | State → district → station, full history or any date range, 30-day mean, 7/14/30/60/90-day forecast with 80% range |
| Districts | Ranking by depth / trend / monsoon recharge, seasonal curves per district |
| Features & seasonality | Seasonal swing, pre- vs post-monsoon recharge, rainfall vs depth, IMD rainfall 1901–2015, model feature importance |
| Model performance | Hold-out MAE/RMSE vs two baselines, actual vs predicted, error by month and district |
| Data quality | Exactly what was dropped and why, flagged stations |
| Export | CSV downloads |
| Sources & deployment | Official source links, verified coverage, database status, authentication and deployment checklist |

## Data actually used

| File | Content | Used for |
|---|---|---|
| `gwl_tel_6_hourly_bihar-gw_br_1991_2020.zip` | Bihar, 538 stations, 38 districts, 2016-06 → 2020-12 | analysis + model |
| `gwl_tel_6_hourly_punjab-gw_pb_1991_2020.zip` | Punjab, 571 stations, mostly 2020 | analysis + model |
| `gwl_tel_6_hourly_punjab_2021_2025_partial.zip` | Punjab, 208 stations in 8 districts (Amritsar–Gurdaspur), 2021 → Sep 2025; **truncated at Excel's row limit** | analysis + model |
| `gwl_tel_6_hourly_punjab_gw_pb_2026_2030.zip` | Punjab, 1,005 stations, **1 Jan – 26 Oct 2026 (real readings)** | analysis + model |
| `district_wise_rainfall_normal.csv` | monthly/annual normals per district | model inputs (climate normals) |
| `rainfall_india_1901_2015.csv` | IMD subdivision rainfall | context chart |
| `atal_jal_disclosed_2015_2022.csv` | Gujarat, Haryana, Karnataka, MP, Maharashtra, Rajasthan, UP only | **not used** (no Punjab/Bihar rows) |

## Important facts about the data

- Despite its name, the "2026–2030" file ends on **26 Oct 2026**; nothing later exists. It is treated as real, current data.
- The **Punjab 2021–2025 file is partial** (cut at Excel's row limit): only 208 stations / 8 districts. Re-export the full file as CSV to cover all 22 districts. Bihar ends 31 Dec 2020, so Bihar forecasts describe early 2021 and cannot be checked against newer data.
- **Depth convention:** depth below ground in metres, larger = deeper. Bihar stores positive depths; Punjab mixes negative and positive between stations, so the magnitude is used. Readings ≥150 m (999, 958…) are placeholders and are removed.
- Nothing is interpolated. Missing days stay missing.
- Rainfall in the model is **district climate normals**, not measured rainfall for a given day (the supplied rainfall files stop in 2015, before telemetry starts).

## Back-tested accuracy

See `PROJECT_REPORT.md` for the full table. In short, on a held-out final ~5 months, the model has lower average error than "no change" at every horizon, by roughly 37–45% in Bihar and 16–22% in Punjab. Errors grow with horizon.

## Team

- Himanshu — 12518949
- Suraj Kumar Sah — 12519259
- Ekansh Choudhary — 12519312

## Download official all-India groundwater source files

The NWIC/NWDP catalogue lists state-wise CSV resources for telemetry and quarterly manual measurements, including historical resources labelled 1991–2020 and newer resources labelled 2021–2025. The catalogue is updated over time; actual dates and district coverage must be checked from each file. Sources:
- Telemetry catalogue: https://nwdp.nwic.gov.in/en/dataset/ground-water-level-telemetry-daily-cgwb-as-assam
- Quarterly manual catalogue: https://nwdp.nwic.gov.in/dataset/gwl-manual-quarterly-central-ground-water-board-department

To download the available official CSV resources on a machine with internet access:

```powershell
python -m pip install -r requirements.txt
python download_official_datasets.py --output data/raw/official
```

To also download resources labelled 2026 onwards, use `--include-current`. The script records source URLs, file names, periods and download status in `download_manifest.json` and `download_manifest.csv`. This is a source-acquisition step only: it does not silently merge different measurement types or automatically retrain the current Punjab/Bihar models. Validate schemas, units, station identities, and state/district coverage before using any additional state in training.

### Normalize and audit the downloaded source files

After downloading the CSV resources, run:

```powershell
python normalize_official_datasets.py --input data/raw/official --output data/processed/official
python audit_downloaded_sources.py --input data/raw/official --output data/processed/official_source_inventory.csv
```

The normalizer processes large files in chunks and writes `normalized_observations.csv`, `normalization_audit.csv`, and `normalization_summary.json`. It preserves the original numeric source value, records source file/URL, and labels telemetry versus quarterly manual observations. It deliberately does **not** convert sign conventions, impute missing groundwater values, or silently combine observation frequencies. Files without confidently detected date/value columns are marked `needs_manual_mapping`. Review each source's unit, sign convention, state/district coverage, and semantics before modeling. The normalized file is an ingestion staging dataset, not yet a certified all-India training dataset.

## Database and refresh options (new)

The project now includes `database_store.py`, which stores normalized official groundwater observations and sync history in a database:

- **Local/offline development:** SQLite database at `data/groundwater_intelligence.sqlite3` (no account or credentials required).
- **Persistent cloud database:** PostgreSQL, including a Supabase Postgres database, by setting `DATABASE_URL` as a private environment variable / Streamlit secret. Do not commit the URL or password to GitHub.
- **Idempotent sync:** records are fingerprinted by source provenance, station, timestamp, measurement type and raw value, so rerunning the import does not duplicate the same source row. Invalid date/value rows are skipped and counted.
- **One-command PowerShell workflow:** from the `Codebase` directory, run `./sync_official_all_india.ps1`. It discovers/downloads official NWIC resources, normalizes files conservatively, audits sources, then syncs the normalized staging CSV to the configured database.
- **Manual database sync:** `python database_store.py --input data/processed/official/normalized_observations.csv`.

The Sources & Deployment tab reports the database mode, row count and last successful sync. It offers a database sync button once the normalized official CSV exists.

**Important:** database storage is not the same as a live official feed. The dashboard's existing forecasting charts still use the bundled Punjab/Bihar processed snapshot. For automated refresh, run the PowerShell workflow on a trusted always-on machine with Windows Task Scheduler (for example, daily) or configure a scheduled cloud job. A refresh only adds observations actually published by the source. Public cloud SQLite files may be ephemeral; use PostgreSQL for durable cloud storage.

### Schedule refresh on Windows

1. Open **Task Scheduler** → **Create Basic Task**.
2. Set a schedule such as daily at 6:00 AM (choose a cadence suitable for your data source).
3. Set the program to `powershell.exe`.
4. Set arguments to `-ExecutionPolicy Bypass -File "C:\path\to\Codebase\sync_official_all_india.ps1"`.
5. Set **Start in** to the Codebase folder. Test the task manually and review the generated download/normalization manifests.

The official catalogue may change and downloads can fail; inspect manifests and coverage reports before claiming all-India completeness or retraining models.

## Authentication and deployment

The dashboard supports Supabase email/password authentication when `SUPABASE_URL` and `SUPABASE_ANON_KEY` are configured. The database can independently use `DATABASE_URL` for PostgreSQL persistence. Copy `.streamlit/secrets.toml.example` as a reference; add real values only in Streamlit Cloud Secrets or local `.streamlit/secrets.toml` (never commit real credentials). Without credentials, the app clearly labels itself as demo mode so local review remains possible.

### Deploy on Streamlit Community Cloud

1. Push the `Codebase/` folder contents to a GitHub repository.
2. In Streamlit Community Cloud, create an app using `app.py` as the main file.
3. Add `SUPABASE_URL` and `SUPABASE_ANON_KEY` in the app's Secrets settings to enable login.
4. Configure Supabase Auth email confirmation, password reset and allowed redirect URLs for the deployed domain.
5. Wait for dependencies to install, open the deployed app, test login/logout, test every dashboard tab and download an export.
6. Run `python tests/smoke_test.py` in CI or a local environment with dependencies installed.

### Coverage and live-data honesty

The current bundled observations and trained models cover **Punjab and Bihar**, not all of India. All-India coverage is the target expansion. New state datasets must be downloaded from official sources, inspected, mapped to a consistent schema, validated and reprocessed before they can be included in the model. The dashboard's refresh button clears cached data and reloads the bundled snapshot; it is **not** a live NWIC/CGWB data fetch. Do not describe the app as live-data-enabled until a verified source endpoint and scheduled ingestion process are configured and tested.

Official references:
- National Water Data Portal: https://nwdp.nwic.gov.in/
- CGWB groundwater-level monitoring: https://cgwb.gov.in/hi/ground-water-level-monitoring
- NWIC quarterly manual groundwater dataset: https://nwdp.nwic.gov.in/dataset/gwl-manual-quarterly-central-ground-water-board-department
- NWIC telemetry groundwater dataset: https://nwdp.nwic.gov.in/dataset/ground-water-level-telemetry-hourly-central-ground-water-board-cgwb
- Open-Meteo historical weather API: https://open-meteo.com/en/docs/historical-weather-api

## Validation status for this package

Python source files pass syntax compilation. A full Streamlit UI smoke test still needs to run in an environment where all `requirements.txt` dependencies can be installed; this working environment had no network access and lacked Streamlit/PyArrow, so runtime rendering could not be verified here.


## Added screening feature

The station and district views include a **relative groundwater watch indicator**. It compares depth against other usable stations or districts in the same state and combines this with the observed depth trend. Labels are "Higher watch", "Monitor", and "Lower relative watch". This is a transparent comparative screening aid only; it is **not** an official CGWB risk category, a regulatory threshold, or a drinking-water safety assessment.

## Current scope and honest limitations

- The bundled model and verified processed observations cover Punjab and Bihar only; all-India is a future expansion target.
- Bihar telemetry in this bundle ends in December 2020. Punjab coverage is uneven across districts and years.
- Rainfall model inputs are district climate normals, not live or station-level daily rainfall.
- Forecast intervals and back-test metrics describe the supplied historical hold-out data; they are not guarantees of future accuracy.
- Supabase login requires the owner to configure Supabase URL/key in deployment secrets. Without credentials, local demo mode remains available.
- SQLite is the default local database. PostgreSQL requires a valid private `DATABASE_URL` and the `psycopg` dependency.
