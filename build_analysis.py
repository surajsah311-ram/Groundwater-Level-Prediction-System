"""P17 Groundwater - step 3: precompute the analysis tables used by the dashboard.

  seasonal.parquet     month-of-year depth anomaly per state/district
  recharge.parquet     pre-monsoon vs post-monsoon depth per station and year
  trends.parquet       per-station multi-year depth trend (m/year, + = water table falling)
  station_summary.parquet / district_summary.parquet   latest level, trend, recharge, rainfall normals
  rain_relation.json   rank correlations between district rainfall normals and depth / trend / recharge
"""
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import p17lib as L

PRE = ((4, 1), (6, 15))        # pre-monsoon window  1 Apr - 15 Jun
POST = ((10, 1), (11, 15))     # post-monsoon window 1 Oct - 15 Nov
LATEST_DAYS = 30


def window_median(d: pd.DataFrame, win, min_days=8):
    (m0, d0), (m1, d1) = win
    md = d["Date"].dt.month * 100 + d["Date"].dt.day
    x = d[(md >= m0 * 100 + d0) & (md <= m1 * 100 + d1)]
    g = x.groupby(["State", "Station", x["Date"].dt.year.rename("Year")], observed=True)["Depth_m"]
    out = g.agg(["median", "size"]).reset_index()
    return out[out["size"] >= min_days].drop(columns="size")


def main():
    d = pd.read_parquet(L.PROC / "daily.parquet")
    st = pd.read_parquet(L.PROC / "stations.parquet")
    use = st[st.usable][["State", "Station"]]
    d = d.merge(use, on=["State", "Station"])
    d["State"] = d["State"].astype(str)
    d["Station"] = d["Station"].astype(str)
    d["District"] = d["District"].astype(str)
    d["Year"] = d["Date"].dt.year

    # ---- seasonal anomaly: depth minus that station's mean in the same (near-complete) year
    ny = d.groupby(["Station", "Year"])["Depth_m"].agg(["mean", "size"]).reset_index()
    ny = ny[ny["size"] >= 250].rename(columns={"mean": "yr_mean"})
    s = d.merge(ny[["Station", "Year", "yr_mean"]], on=["Station", "Year"])
    s["Anomaly_m"] = s["Depth_m"] - s["yr_mean"]
    s["Month"] = s["Date"].dt.month
    seasonal = s.groupby(["State", "District", "Month"], as_index=False).agg(
        Anomaly_m=("Anomaly_m", "mean"), stations=("Station", "nunique"))
    seasonal.to_parquet(L.PROC / "seasonal.parquet", index=False)

    # ---- monsoon recharge
    pre = window_median(d, PRE).rename(columns={"median": "PreMonsoon_m"})
    post = window_median(d, POST).rename(columns={"median": "PostMonsoon_m"})
    rc = pre.merge(post, on=["State", "Station", "Year"])
    rc["Recharge_m"] = rc["PreMonsoon_m"] - rc["PostMonsoon_m"]     # + = water table rose
    rc = rc.merge(st[["Station", "District"]], on="Station")
    rc.to_parquet(L.PROC / "recharge.parquet", index=False)

    # ---- multi-year trend (same calendar window each year, so seasons do not bias it)
    tr = []
    b = d[(d.State == "Bihar")]
    bn = b.groupby(["Station", "Year"])["Depth_m"].agg(["median", "size"]).reset_index()
    bn = bn[bn["size"] >= 300].pivot(index="Station", columns="Year", values="median")
    if {2018, 2020} <= set(bn.columns):
        t = ((bn[2020] - bn[2018]) / 2.0).dropna().rename("Trend_m_per_yr").reset_index()
        t["State"], t["Basis"] = "Bihar", "median depth 2020 vs 2018 (full years)"
        tr.append(t)
    p = d[(d.State == "Punjab") & d["Date"].dt.dayofyear.between(60, 299)]
    pn = p.groupby(["Station", "Year"])["Depth_m"].agg(["median", "size"]).reset_index()
    pn = pn[pn["size"] >= 150].pivot(index="Station", columns="Year", values="median")
    if {2020, 2026} <= set(pn.columns):
        t = ((pn[2026] - pn[2020]) / 6.0).dropna().rename("Trend_m_per_yr").reset_index()
        t["State"], t["Basis"] = "Punjab", "median depth 2026 vs 2020, same days of year (Mar-Oct)"
        tr.append(t)
    trends = pd.concat(tr, ignore_index=True)
    trends = trends[trends["Trend_m_per_yr"].abs() < 8]          # drop obviously broken pairs
    trends.to_parquet(L.PROC / "trends.parquet", index=False)

    # ---- latest level per station (last 30 days of that station's data)
    last = d.groupby("Station")["Date"].transform("max")
    lat30 = d[d["Date"] > last - pd.Timedelta(days=LATEST_DAYS)]
    latest = lat30.groupby("Station").agg(Latest_m=("Depth_m", "median"), Latest_date=("Date", "max")).reset_index()
    ss = (st[st.usable].merge(latest, on="Station", how="left")
          .merge(trends[["Station", "Trend_m_per_yr"]], on="Station", how="left"))
    rec = rc.groupby("Station")["Recharge_m"].mean().rename("Recharge_m").reset_index()
    ss = ss.merge(rec, on="Station", how="left")
    ss.to_parquet(L.PROC / "station_summary.parquet", index=False)

    # ---- district summary + rainfall normals
    normals = L.load_rain_normals()
    ds = ss.groupby(["State", "District"]).agg(
        Stations=("Station", "size"), Latest_m=("Latest_m", "median"),
        Trend_m_per_yr=("Trend_m_per_yr", "median"), Falling_share=("Trend_m_per_yr", lambda x: float((x.dropna() > 0).mean()) if x.notna().any() else np.nan),
        Recharge_m=("Recharge_m", "median"), Latest_date=("Latest_date", "max")).reset_index()
    ds["Annual_rain_mm"] = [float(normals.loc[L.rain_key(x), "ANNUAL"]) for x in ds["District"]]
    ds["Monsoon_rain_mm"] = [float(normals.loc[L.rain_key(x), "Jun-Sep"]) for x in ds["District"]]
    ds.to_parquet(L.PROC / "district_summary.parquet", index=False)

    rel = {}
    for state, g in ds.groupby("State"):
        r = {}
        for y in ["Latest_m", "Trend_m_per_yr", "Recharge_m"]:
            gg = g[["Annual_rain_mm", y]].dropna()
            if len(gg) >= 8:
                rho, pv = spearmanr(gg["Annual_rain_mm"], gg[y])
                r[f"rain_vs_{y}"] = {"rho": float(rho), "p": float(pv), "n_districts": int(len(gg))}
        rel[state] = r
    (L.PROC / "rain_relation.json").write_text(json.dumps(rel, indent=2))
    print(ds.round(2).to_string())
    print(json.dumps(rel, indent=2))


if __name__ == "__main__":
    main()
