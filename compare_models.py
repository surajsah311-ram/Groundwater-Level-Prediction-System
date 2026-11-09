"""P17 Groundwater - optional step 4: compare the gradient-boosting model with other model families.

Same features, same chronological hold-out as train_models.py, horizons 7/30/90 days.
Ridge regression and Random Forest are trained on a random sample of the training rows (speed);
the gradient-boosting numbers are read from backtest_metrics.csv (full training set).
Output: data/processed/model_comparison.csv
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import p17lib as L
from train_models import HOLDOUT_DAYS, build_state_table

HORIZONS = [7, 30, 90]
SAMPLE = 200_000


def main():
    daily = pd.read_parquet(L.PROC / "daily.parquet")
    stations = pd.read_parquet(L.PROC / "stations.parquet")
    normals = L.load_rain_normals()
    bt = pd.read_csv(L.PROC / "backtest_metrics.csv").set_index(["State", "Horizon_days"])
    rows = []
    for state in sorted(stations.State.unique()):
        T = build_state_table(daily, stations, state, normals)
        cutoff = T["Date"].max() - pd.Timedelta(days=HOLDOUT_DAYS)
        for h in HORIZONS:
            D = T[T[f"y_{h}"].notna()].reset_index(drop=True)
            train = D[D["Date"] + pd.Timedelta(days=h) < cutoff]
            test = D[D["Date"] >= cutoff]
            tr = train.sample(min(SAMPLE, len(train)), random_state=0)
            Xtr, ytr = L.design_matrix(tr, h), tr[f"y_{h}"]
            Xte, yte = L.design_matrix(test, h), test[f"y_{h}"].to_numpy()
            models = {
                "Ridge regression": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1.0)),
                "Random Forest": make_pipeline(SimpleImputer(strategy="median"),
                                               RandomForestRegressor(n_estimators=100, max_depth=14, min_samples_leaf=50,
                                                                     n_jobs=-1, random_state=0)),
            }
            out = {"Persistence (no change)": np.zeros(len(test))}
            for name, m in models.items():
                out[name] = m.fit(Xtr, ytr).predict(Xte)
            for name, p in out.items():
                e = p - yte
                rows.append({"State": state, "Horizon_days": h, "Model": name, "MAE_m": np.mean(np.abs(e)),
                             "RMSE_m": float(np.sqrt(np.mean(e ** 2))), "Within_0.5m": float(np.mean(np.abs(e) <= 0.5))})
            b = bt.loc[(state, h)]
            rows.append({"State": state, "Horizon_days": h, "Model": "Gradient boosting (P17)", "MAE_m": b["MAE_model"],
                         "RMSE_m": b["RMSE_model"], "Within_0.5m": b["Within_0.5m_model"]})
            print(state, h, "done", flush=True)
    pd.DataFrame(rows).to_csv(L.PROC / "model_comparison.csv", index=False)


if __name__ == "__main__":
    main()
