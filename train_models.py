"""P17 Groundwater - step 2: backtest and train the forecasting models.

For every state and horizon h in {7,14,30,60,90} days a gradient-boosting model predicts
    change = depth(t+h) - current 7-day-mean depth
from the station's recent history, season, location and district rainfall normals.

Honest evaluation: a chronological hold-out. Everything with a target date before the cutoff
(last observed date - 150 days) is training data; origins after the cutoff are the test set.
Models are compared with two baselines (persistence and a seasonal-average change).
Final models used by the dashboard are then re-fitted on all available data.
"""
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance

import p17lib as L

HOLDOUT_DAYS = 150
PARAMS = dict(loss="absolute_error", max_iter=250, learning_rate=0.08, max_leaf_nodes=31,
              min_samples_leaf=200, l2_regularization=1.0, random_state=42)


def build_state_table(daily: pd.DataFrame, stations: pd.DataFrame, state: str, normals) -> pd.DataFrame:
    st = stations[(stations.State == state) & stations.usable]
    d = daily[(daily.State == state) & daily.Station.isin(st.Station)]
    d = d.sort_values(["Station", "Date"])
    meta = st.set_index("Station")
    parts = []
    rain_cache = {}
    for station, g in d.groupby("Station", observed=True):
        m = meta.loc[station]
        if m.District not in rain_cache:
            rain_cache[m.District] = L.rain_static(m.District, normals)
        ann, jjas, cum = rain_cache[m.District]
        dates, depth = g["Date"].to_numpy(), g["Depth_m"].to_numpy(dtype=float)
        for seg in L.split_segments(dates):
            if len(seg) < 30:
                continue
            f = L.segment_features(dates[seg], depth[seg], float(m.Latitude), float(m.Longitude), ann, jjas, cum)
            f["Station"] = station
            f["District"] = m.District
            parts.append(f)
    return pd.concat(parts, ignore_index=True)


def main():
    t0 = time.time()
    daily = pd.read_parquet(L.PROC / "daily.parquet")
    stations = pd.read_parquet(L.PROC / "stations.parquet")
    normals = L.load_rain_normals()
    L.MODELS.mkdir(exist_ok=True)

    metrics, preds, imps, by_district, by_month = [], [], [], [], []
    interval, summary = {}, {}
    for state in sorted(stations.State.unique()):
        print(f"\n=== {state}: building features", flush=True)
        T = build_state_table(daily, stations, state, normals)
        last = T["Date"].max()
        cutoff = last - pd.Timedelta(days=HOLDOUT_DAYS)
        summary[state] = {"feature_rows": int(len(T)), "stations": int(T.Station.nunique()),
                          "last_date": str(last.date()), "cutoff_date": str(cutoff.date())}
        print(f"   {len(T):,} origin rows, {T.Station.nunique()} stations, hold-out starts {cutoff.date()}", flush=True)
        interval[state] = {}
        for h in L.HORIZONS:
            ok = T[f"y_{h}"].notna()
            D = T[ok].reset_index(drop=True)
            tgt_date = D["Date"] + pd.Timedelta(days=h)
            train = D[tgt_date < cutoff]          # target already known before the cutoff
            test = D[D["Date"] >= cutoff]         # origins after the cutoff
            Xtr, ytr = L.design_matrix(train, h), train[f"y_{h}"]
            Xte, yte = L.design_matrix(test, h), test[f"y_{h}"]
            model = HistGradientBoostingRegressor(**PARAMS).fit(Xtr, ytr)
            p = model.predict(Xte)
            # baselines
            persist = np.zeros(len(test))
            month_delta = train.assign(m=train["Date"].dt.month).groupby("m")[f"y_{h}"].mean()
            seasonal = test["Date"].dt.month.map(month_delta).fillna(0.0).to_numpy()
            err = {"model": p - yte.to_numpy(), "persistence": persist - yte.to_numpy(),
                   "seasonal_avg_change": seasonal - yte.to_numpy()}
            row = {"State": state, "Horizon_days": h, "n_train": len(train), "n_test": len(test),
                   "test_stations": int(test.Station.nunique())}
            for k, e in err.items():
                row[f"MAE_{k}"] = float(np.mean(np.abs(e)))
                row[f"RMSE_{k}"] = float(np.sqrt(np.mean(e ** 2)))
            row["MedAE_model"] = float(np.median(np.abs(err["model"])))
            row["Within_0.5m_model"] = float(np.mean(np.abs(err["model"]) <= 0.5))
            row["Within_0.5m_persistence"] = float(np.mean(np.abs(err["persistence"]) <= 0.5))
            row["Skill_vs_persistence_MAE"] = 1 - row["MAE_model"] / row["MAE_persistence"]
            res = -err["model"]            # actual - predicted
            q10, q90 = np.quantile(res, [0.10, 0.90])
            interval[state][str(h)] = {"q10": float(q10), "q90": float(q90)}
            row["Interval80_coverage"] = float(np.mean((res >= q10) & (res <= q90)))
            metrics.append(row)
            print(f"   h={h:>2}: MAE model {row['MAE_model']:.3f} | persistence {row['MAE_persistence']:.3f} "
                  f"| seasonal {row['MAE_seasonal_avg_change']:.3f} | skill {row['Skill_vs_persistence_MAE']:+.1%}", flush=True)

            # keep a sample of test predictions (levels, not deltas) for plots
            take = test.sample(min(6000, len(test)), random_state=1)
            pp = model.predict(L.design_matrix(take, h))
            preds.append(pd.DataFrame({"State": state, "Horizon_days": h, "Station": take.Station.to_numpy(),
                                       "District": take.District.to_numpy(), "Origin": take.Date.to_numpy(),
                                       "Target_date": (take.Date + pd.Timedelta(days=h)).to_numpy(),
                                       "Current_m": take.cur.to_numpy(), "Actual_m": take.cur.to_numpy() + take[f"y_{h}"].to_numpy(),
                                       "Predicted_m": take.cur.to_numpy() + pp,
                                       "Persistence_m": take.cur.to_numpy()}))
            # per-district / per-month errors on the full test set
            tt = test.assign(abs_err_model=np.abs(err["model"]), abs_err_persist=np.abs(err["persistence"]))
            by_district.append(tt.groupby("District").agg(n=("abs_err_model", "size"),
                MAE_model=("abs_err_model", "mean"), MAE_persistence=("abs_err_persist", "mean"))
                .reset_index().assign(State=state, Horizon_days=h))
            by_month.append(tt.assign(Month=tt.Date.dt.month).groupby("Month").agg(n=("abs_err_model", "size"),
                MAE_model=("abs_err_model", "mean"), MAE_persistence=("abs_err_persist", "mean"))
                .reset_index().assign(State=state, Horizon_days=h))

            # permutation importance on a test sample (increase in MAE when a feature is shuffled)
            samp = test.sample(min(8000, len(test)), random_state=2)
            pi = permutation_importance(model, L.design_matrix(samp, h), samp[f"y_{h}"], n_repeats=3,
                                        random_state=0, scoring="neg_mean_absolute_error", n_jobs=1)
            imps.append(pd.DataFrame({"State": state, "Horizon_days": h, "Feature": L.FEATURES,
                                      "MAE_increase_m": pi.importances_mean}))

            # final model: all rows whose target is observed
            final = HistGradientBoostingRegressor(**PARAMS).fit(L.design_matrix(D, h), D[f"y_{h}"])
            joblib.dump(final, L.MODELS / f"{state.lower()}_h{h}.joblib", compress=3)

    pd.DataFrame(metrics).to_csv(L.PROC / "backtest_metrics.csv", index=False)
    pd.concat(preds).to_parquet(L.PROC / "backtest_predictions.parquet", index=False)
    pd.concat(imps).to_csv(L.PROC / "feature_importance.csv", index=False)
    pd.concat(by_district).to_csv(L.PROC / "backtest_by_district.csv", index=False)
    pd.concat(by_month).to_csv(L.PROC / "backtest_by_month.csv", index=False)
    meta = {"states": sorted(stations.State.unique().tolist()), "horizons": L.HORIZONS,
            "features": L.FEATURES, "holdout_days": HOLDOUT_DAYS, "interval": interval,
            "params": {k: v for k, v in PARAMS.items()}, "summary": summary,
            "target": "depth(t+h) - 7-day-mean depth at origin t (metres below ground)"}
    (L.MODELS / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nTraining finished in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
