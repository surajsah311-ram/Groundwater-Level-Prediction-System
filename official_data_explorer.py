"""Official groundwater data explorer for the Ground Water Level Prediction System.

Shows normalized official source observations separately from the bundled model dataset.
It does not imply that downloaded source data has been validated for model training.
"""
from pathlib import Path
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
OFFICIAL_DIR = ROOT / "data" / "processed" / "official"
NORMALIZED = OFFICIAL_DIR / "normalized_observations.csv"

def render_official_data_explorer(tab):
    with tab:
        st.subheader("Official Groundwater Data Explorer")
        st.info("Official downloaded observations are displayed separately from the Punjab/Bihar model dashboard. Review units, measurement type, date coverage and source provenance before using them for model training.")
        if not NORMALIZED.exists():
            st.warning("No normalized official dataset found yet.")
            st.write("Expected file:")
            st.code(str(NORMALIZED))
            st.markdown("**Next steps (PowerShell, from the Codebase folder):**")
            st.code("python download_official_datasets.py --output data/raw/official\npython normalize_official_datasets.py --input data/raw/official --output data/processed/official", language="powershell")
            st.caption("The download step requires internet access. Some portal resources may fail or need manual review.")
        else:
            st.success(f"Normalized official dataset found: {NORMALIZED.name}")
            try:
                df = pd.read_csv(NORMALIZED, low_memory=False)
                st.metric("Normalized observation rows", f"{len(df):,}")
                c1, c2 = st.columns(2)
                state_col = next((c for c in ["state", "State"] if c in df.columns), None)
                type_col = next((c for c in ["measurement_type", "Measurement_Type"] if c in df.columns), None)
                if state_col:
                    states = sorted(df[state_col].dropna().astype(str).unique().tolist())
                    selected = c1.multiselect("State filter", states, default=states[:min(5, len(states))])
                    if selected:
                        df = df[df[state_col].astype(str).isin(selected)]
                if type_col:
                    types = sorted(df[type_col].dropna().astype(str).unique().tolist())
                    selected_types = c2.multiselect("Measurement type", types, default=types)
                    if selected_types:
                        df = df[df[type_col].astype(str).isin(selected_types)]
                st.caption(f"Showing {len(df):,} rows from the normalized staging file.")
                st.dataframe(df.head(5000), use_container_width=True, hide_index=True)
                st.download_button("Download filtered official data (CSV)", df.to_csv(index=False).encode("utf-8"), file_name="official_groundwater_filtered.csv", mime="text/csv")
                st.caption("Maximum 5,000 rows are displayed in the table; the download contains all filtered rows.")
            except Exception as exc:
                st.error(f"Could not read normalized official data: {exc}")
        st.divider()
        st.markdown("**Database status**")
        try:
            from database_store import database_status
            status = database_status()
            a, b, c = st.columns(3)
            a.metric("Database mode", status.get("mode", "Unknown"))
            b.metric("Stored observations", f"{status.get('observation_rows', 0):,}")
            c.metric("Last successful sync", status.get("last_sync_utc") or "Never")
        except Exception as exc:
            st.warning(f"Database status unavailable: {exc}")
