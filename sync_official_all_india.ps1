$ErrorActionPreference = 'Stop'
# Run from the Codebase folder. Downloads official NWIC resources, normalizes them conservatively,
# audits source files, then syncs normalized observations into local SQLite or DATABASE_URL PostgreSQL.
python -m pip install -r requirements.txt
python download_official_datasets.py --output data/raw/official --include-current
python normalize_official_datasets.py --input data/raw/official --output data/processed/official
python audit_downloaded_sources.py --input data/raw/official --output data/processed/official_source_inventory.csv
python database_store.py --input data/processed/official/normalized_observations.csv
