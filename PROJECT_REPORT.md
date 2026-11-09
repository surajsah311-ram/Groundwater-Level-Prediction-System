# P17 Groundwater Prediction — Back-test Report

## Scope

This report summarizes the bundled hold-out metrics in `data/processed/backtest_metrics.csv`. The trained models and processed observations currently cover **Punjab and Bihar only**. Forecast horizons are 7, 14, 30, 60 and 90 days.

The target is the change in depth below ground from the 7-day mean at the forecast origin to the observed depth at the target horizon. A positive depth change means the water table is deeper. The comparison baselines are persistence (no change) and a seasonal average-change baseline.

## Hold-out mean absolute error

Lower MAE is better. Values are metres.

| State | Horizon (days) | Model MAE | Persistence MAE | Skill vs persistence |
|---|---:|---:|---:|---:|
| Bihar | 7 | 0.169 | 0.288 | 41.3% |
| Bihar | 14 | 0.273 | 0.459 | 40.5% |
| Bihar | 30 | 0.431 | 0.768 | 43.9% |
| Bihar | 60 | 0.654 | 1.191 | 45.1% |
| Bihar | 90 | 0.876 | 1.401 | 37.4% |
| Punjab | 7 | 0.255 | 0.302 | 15.7% |
| Punjab | 14 | 0.363 | 0.436 | 16.8% |
| Punjab | 30 | 0.578 | 0.701 | 17.5% |
| Punjab | 60 | 0.842 | 1.047 | 19.6% |
| Punjab | 90 | 1.138 | 1.463 | 22.2% |

## Interpretation and limitations

- In the bundled hold-out evaluation, the model's MAE is lower than persistence for every reported state and horizon.
- Error generally increases with forecast horizon.
- The nominal 80% prediction intervals show approximately 80% empirical coverage in this hold-out dataset; this does not guarantee future coverage.
- These results are specific to the supplied hold-out periods and stations. They are not a guarantee of performance in other years or locations.
- Groundwater changes are also affected by pumping, local geology, irrigation and rainfall. The model uses district rainfall normals as climate features, not live daily rainfall or pumping data.
- Bihar's supplied telemetry ends on 31 December 2020, so its forecasts cannot be validated against recent Bihar observations in this package.
- Punjab's 2021–2025 source export is partial; coverage for that period is incomplete.
- This package is not an all-India or live official monitoring service. Current bundled model/data coverage is Punjab and Bihar only.

## Reproducibility

The source metrics are stored in `data/processed/backtest_metrics.csv`; model metadata and forecast interval quantiles are stored in `models/meta.json`. The dashboard is launched from the `Codebase` directory with:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```


## Relative groundwater watch indicator (dashboard enhancement)

The station explorer now reports the station's percentile within the usable stations in its state and an approximate recent 90-day depth trend. The district view also classifies districts relative to other districts in the selected state. Labels are "Higher watch", "Monitor", and "Lower relative watch". These labels are intended only for comparative screening and are not official CGWB risk classes, regulatory limits, or a drinking-water safety assessment.

## Scope and limitations to state during the demo

- Verified observations and trained forecasts in this version cover Punjab and Bihar, not all of India.
- Punjab coverage varies by year and district; Bihar's supplied telemetry ends on 31 December 2020.
- Rainfall model features use district climate normals rather than live or daily measured rainfall.
- Forecast accuracy is reported from a historical hold-out period and should not be presented as a guarantee.
