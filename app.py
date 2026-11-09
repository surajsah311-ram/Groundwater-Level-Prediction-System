"""Ground Water Level Prediction System (Punjab & Bihar). Streamlit dashboard.

Run:  python -m streamlit run app.py
Everything shown here is computed from the supplied telemetry files by prepare_data.py,
train_models.py and build_analysis.py.  Depth is metres BELOW GROUND: bigger = deeper = worse.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import auth
import p17lib as L

PROC = L.PROC
st.set_page_config(page_title="Ground Water Level Prediction System", page_icon="💧", layout="wide")
if not auth.require_auth():
    st.stop()
st.markdown(
    """<style>
.block-container{padding-top:1.2rem;padding-bottom:2rem}
.hero{padding:1.1rem 1.4rem;border-radius:14px;background:linear-gradient(120deg,#0b2545,#155e75);color:#fff;margin-bottom:1rem}
.hero h1{margin:0;color:#fff;font-size:1.7rem}.hero p{margin:.35rem 0 0;color:#d8f3ff;font-size:.95rem}
.note{font-size:.85rem;color:#64748b}
</style>""",
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="hero"><h1>💧 Groundwater Level Prediction System</h1>'
    "<p>Groundwater intelligence · verified coverage currently Punjab &amp; Bihar · daily depth-to-water · "
    "district analysis · rainfall &amp; seasonal features · tested 7–90 day forecasts</p></div>",
    unsafe_allow_html=True,
)

needed = ["daily.parquet", "stations.parquet", "station_summary.parquet", "district_summary.parquet"]
if not all((PROC / n).exists() for n in needed):
    st.error("Processed data not found. Run, in order: prepare_data.py, train_models.py, build_analysis.py")
    st.code("python prepare_data.py\npython train_models.py\npython build_analysis.py", language="bash")
    st.stop()


# ------------------------------------------------------------------ cached loaders
@st.cache_data(show_spinner="Loading daily records...")
def load_daily():
    d = pd.read_parquet(PROC / "daily.parquet")
    for c in ["State", "District", "Station"]:
        d[c] = d[c].astype(str)
    return d


@st.cache_data
def load(name):
    p = PROC / name
    if not p.exists():
        return None
    return pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)


@st.cache_data
def load_json(name):
    p = PROC / name
    return json.loads(p.read_text()) if p.exists() else {}


def break_gaps(df, col="Depth_m", gap_days=7):
    """Insert a NaN row after any gap longer than gap_days so lines are not drawn across missing periods."""
    d = df[["Date", col]].reset_index(drop=True)
    big = d["Date"].diff().dt.days > gap_days
    if not big.any():
        return d
    fill = pd.DataFrame({"Date": d.loc[big, "Date"] - pd.Timedelta(days=1), col: np.nan})
    return pd.concat([d, fill]).sort_values("Date").reset_index(drop=True)


def rnd(df, n=3):
    """Round only the float columns (DataFrame.round warns on datetime columns)."""
    return df.round({c: n for c in df.select_dtypes("float").columns})


@st.cache_data(show_spinner=False)
def state_csv(state):
    return rnd(daily[daily.State == state]).to_csv(index=False, date_format="%Y-%m-%d").encode()


@st.cache_resource
def get_models():
    return L.load_models()


@st.cache_data
def india_rain():
    p = L.RAW / "rainfall_india_1901_2015.csv"
    if not p.exists():
        return None
    r = pd.read_csv(p)
    return r[r["SUBDIVISION"].isin(["PUNJAB", "BIHAR"])][["SUBDIVISION", "YEAR", "ANNUAL", "Jun-Sep"]]


daily = load_daily()
stations = load("stations.parquet")
sumry = load("station_summary.parquet")
dist = load("district_summary.parquet")
seasonal = load("seasonal.parquet")
recharge = load("recharge.parquet")
trends = load("trends.parquet")
metrics = load("backtest_metrics.csv")
preds = load("backtest_predictions.parquet")
imps = load("feature_importance.csv")
by_dist = load("backtest_by_district.csv")
by_month = load("backtest_by_month.csv")
qrep = load_json("quality_report.json")
rel = load_json("rain_relation.json")
meta, models = get_models()

tabs = st.tabs(["Overview", "Station explorer & forecast", "Districts", "Features & seasonality",
                "Model performance", "Data quality", "Export", "Sources & deployment"])

# ====================================================================== Overview
with tabs[0]:
    fin = qrep.get("final", {})
    c = st.columns(5)
    c[0].metric("States", len(fin.get("states", [])))
    c[1].metric("Districts", fin.get("districts", "–"))
    c[2].metric("Stations", f"{fin.get('stations', 0):,}")
    c[3].metric("Daily station records", f"{fin.get('station_days', 0):,}")
    c[4].metric("Date coverage", f"{fin.get('first_date', '')[:4]} – {fin.get('last_date', '')[:4]}")
    st.caption(
        "Depth = metres below ground (larger = deeper water table). Each record is the median of that day's valid "
        "telemetry readings. Missing days stay missing."
    )
    cs = st.columns([1.2, 1])
    with cs[0]:
        sel_state = st.radio("State", ["Punjab", "Bihar"], horizontal=True, key="ov_state")
        m = sumry[(sumry.State == sel_state) & sumry.Latest_m.notna()].copy()
        m["Latest depth (m)"] = m["Latest_m"].round(2)
        fig = px.scatter(
            m, x="Longitude", y="Latitude", color="Latest depth (m)", hover_name="StationLabel",
            hover_data={"District": True, "Latitude": ":.3f", "Longitude": ":.3f"},
            color_continuous_scale="YlGnBu", range_color=[0, float(m["Latest_m"].quantile(0.97))], height=520,
        )
        fig.update_traces(marker=dict(size=7, line=dict(width=0.3, color="#333")))
        fig.update_yaxes(scaleanchor="x", scaleratio=1.0, title="Latitude")
        fig.update_xaxes(title="Longitude")
        fig.update_layout(margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, width="stretch")
        last_dates = m["Latest_date"].max()
        st.caption(f"Colour = median depth over each station's last 30 recorded days (latest data for {sel_state}: "
                   f"{pd.Timestamp(last_dates):%d %b %Y}).")
    with cs[1]:
        rows = []
        for s_, v in fin.get("by_state", {}).items():
            rows.append({"State": s_, "Districts": v["districts"], "Stations": v["stations"],
                         "Station-days": v["station_days"], "From": v["first_date"], "To": v["last_date"]})
        st.markdown("##### Coverage by state")
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        yr = daily.assign(Year=daily["Date"].dt.year).groupby(["State", "Year"]).size().reset_index(name="Records")
        yr = yr[yr.Year >= 2016]
        f2 = px.bar(yr, x="Year", y="Records", color="State", barmode="group", height=300,
                    title="Daily station records per year")
        st.plotly_chart(f2, width="stretch")
        st.info("Punjab 2021–2025 comes from a partial file (cut at Excel's row limit): only 208 stations in 8 districts "
                "(Amritsar–Gurdaspur) up to Sep 2025. The other 14 districts have 2020 and 2026 only. Bihar runs 2016-06 to 2020-12.")

# ====================================================================== Station explorer
with tabs[1]:
    s1, s2, s3 = st.columns(3)
    state = s1.selectbox("State", ["Punjab", "Bihar"], key="se_state")
    sd = stations[stations.State == state]
    district = s2.selectbox("District", sorted(sd.District.unique()), key="se_district")
    show_flagged = s3.checkbox("Show flagged (unreliable) stations", value=False)
    sdd = sd[sd.District == district]
    if not show_flagged:
        sdd = sdd[sdd.usable]
    if sdd.empty:
        st.warning("No usable stations in this district.")
        st.stop()
    sdd = sdd.sort_values("obs_days", ascending=False)
    label_to_id = dict(zip(sdd.StationLabel, sdd.Station))
    label = st.selectbox(f"Monitoring station ({len(sdd)} in {district}; most data first)", list(label_to_id))
    station = label_to_id[label]
    srow = sdd[sdd.Station == station].iloc[0]
    ser = daily[daily.Station == station].sort_values("Date")
    dmin, dmax = ser.Date.min().date(), ser.Date.max().date()

    r1, r2 = st.columns([1, 2])
    rng = r1.radio("Date range", ["Full history", "Last 12 months", "Last 90 days", "Custom"], horizontal=True,
                   index=1 if state == "Punjab" else 0, key=f"rng_{state}")
    if rng == "Full history":
        a, b = dmin, dmax
    elif rng == "Last 12 months":
        a, b = max(dmin, (pd.Timestamp(dmax) - pd.DateOffset(years=1)).date()), dmax
    elif rng == "Last 90 days":
        a, b = max(dmin, (pd.Timestamp(dmax) - pd.Timedelta(days=90)).date()), dmax
    else:
        picked = r2.date_input("From – to", value=(dmin, dmax), min_value=dmin, max_value=dmax)
        a, b = (picked if isinstance(picked, tuple) and len(picked) == 2 else (dmin, dmax))
    view = ser[(ser.Date >= pd.Timestamp(a)) & (ser.Date <= pd.Timestamp(b))]
    if view.empty:
        st.warning("No readings in that range for this station. Widen the date range.")
        st.stop()

    st.caption(f"**{state} > {district} > {label}** · {srow.Block} block · {srow.Latitude:.3f}°N {srow.Longitude:.3f}°E · "
               f"history {dmin:%d %b %Y} -> {dmax:%d %b %Y} · {int(srow.obs_days):,} observed days "
               f"({srow.coverage:.0%} coverage within reporting years)")
    if not bool(srow.usable):
        st.warning("This station is flagged as unreliable (readings jump by tens of metres, sit at placeholder depths "
                   "or never change). It is shown for transparency but not forecast.")
    k = st.columns(5)
    k[0].metric("Latest depth (m)", f"{view.Depth_m.iloc[-1]:.2f}")
    k[1].metric("Mean (m)", f"{view.Depth_m.mean():.2f}")
    k[2].metric("Shallowest (m)", f"{view.Depth_m.min():.2f}")
    k[3].metric("Deepest (m)", f"{view.Depth_m.max():.2f}")
    k[4].metric("Days in view", f"{len(view):,}")

    # Relative screening indicator: compare the station's median depth within its state.
    state_depths = stations.loc[(stations.State == state) & stations.usable, "median_depth"].dropna()
    depth_pct = float((state_depths <= float(srow.median_depth)).mean()) if len(state_depths) else np.nan
    recent = ser[ser.Date >= (ser.Date.max() - pd.Timedelta(days=90))].dropna(subset=["Depth_m"])
    trend_90 = np.nan
    if len(recent) >= 10:
        x_days = (recent.Date - recent.Date.min()).dt.days.to_numpy(dtype=float)
        trend_90 = float(np.polyfit(x_days, recent.Depth_m.to_numpy(dtype=float), 1)[0] * 365.25)
    status = L.relative_groundwater_status(depth_pct, trend_90)
    st.markdown("#### Relative groundwater watch indicator")
    rw1, rw2, rw3 = st.columns(3)
    rw1.metric("State depth percentile", f"{depth_pct:.0%}" if pd.notna(depth_pct) else "Not available")
    rw2.metric("Recent trend (m/year)", f"{trend_90:+.2f}" if pd.notna(trend_90) else "Insufficient data")
    rw3.metric("Screening label", status)
    st.caption("This is a comparative screening indicator, not an official CGWB risk category or a drinking-water/safety assessment. "
               "Higher percentile means deeper-than-most stations in the same state; positive trend means depth is increasing.")

    fig = go.Figure()
    vb = break_gaps(view)
    fig.add_trace(go.Scatter(x=vb.Date, y=vb.Depth_m, mode="lines", name="Daily depth", line=dict(width=1.3)))
    roll = view.set_index("Date")["Depth_m"].rolling("30D", min_periods=7).mean().reset_index()
    rb = break_gaps(roll)
    fig.add_trace(go.Scatter(x=rb.Date, y=rb.Depth_m, mode="lines", name="30-day mean", line=dict(width=2.5, dash="dot")))

    fc = None
    do_fc = bool(srow.usable) and models
    if do_fc:
        res = L.forecast_station(models, meta, state, district, ser.Date.to_numpy(), ser.Depth_m.to_numpy(),
                                 float(srow.Latitude), float(srow.Longitude))
        if res is not None:
            origin, cur, fc = res
            xs = [origin] + fc.Date.tolist()
            fig.add_trace(go.Scatter(x=xs + xs[::-1], y=[cur] + fc.High_m.tolist() + ([cur] + fc.Low_m.tolist())[::-1],
                                     mode="lines", fill="toself", fillcolor="rgba(220,80,60,0.15)", line=dict(width=0),
                                     name="80% range", hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=xs, y=[cur] + fc.Forecast_m.tolist(), mode="lines+markers", name="Forecast",
                                     line=dict(color="#d6452f", width=2.5), marker=dict(size=8)))
    fig.update_yaxes(autorange="reversed", title="Depth below ground (m)")
    fig.update_layout(height=470, hovermode="x unified", xaxis_title="Date", margin=dict(t=30),
                      legend=dict(orientation="h", y=1.08))
    st.plotly_chart(fig, width="stretch")
    st.caption("The depth axis is flipped so a falling water table plots downward. Gaps are real missing days (no interpolation).")

    st.markdown("#### Forecast")
    if fc is None:
        st.info("No forecast for this station." if not do_fc else "Not enough recent data to build a forecast origin.")
    else:
        state_last = pd.Timestamp(daily.loc[daily.State == state, "Date"].max())
        if origin < state_last - pd.Timedelta(days=14):
            st.warning(f"This station's last reading is {origin:%d %b %Y}, earlier than the latest {state} data "
                       f"({state_last:%d %b %Y}); the forecast starts from its own last reading.")
        if state == "Bihar":
            st.warning("Bihar telemetry in the supplied files ends on 31 Dec 2020, so these forecasts describe "
                       "Jan–Mar 2021 and cannot be checked against newer data. Add newer Bihar files to forecast from today.")
        mm = metrics[metrics.State == state].set_index("Horizon_days")
        t = fc.copy()
        t["Change_m"] = t["Forecast_m"] - cur
        t["Typical_error_m"] = [mm.loc[h, "MAE_model"] for h in t.Horizon_days]
        st.write(f"Origin **{origin:%d %b %Y}**, current level (7-day mean) **{cur:.2f} m**. "
                 f"Positive change = water table getting deeper.")
        show = t.assign(Date=t.Date.dt.strftime("%d %b %Y")).rename(columns={
            "Horizon_days": "Days ahead", "Forecast_m": "Forecast depth (m)", "Low_m": "Low (m)", "High_m": "High (m)",
            "Change_m": "Change vs now (m)", "Typical_error_m": f"Typical {state} error, back-tested (m)"})
        st.dataframe(rnd(show, 2), hide_index=True, width="stretch")
        st.download_button("Download forecast CSV", rnd(t).to_csv(index=False, date_format="%Y-%m-%d").encode(),
                           file_name=f"p17_forecast_{station[:30]}.csv", mime="text/csv")
        st.caption("Forecasts are statistical estimates from past depth, season, location and district rainfall *normals* "
                   "(not live rainfall). The range is the 10th–90th percentile of back-test errors for this state and horizon.")

    with st.expander("Station time-series features"):
        f = ser[["Date", "Depth_m"]].copy().set_index("Date")
        feat = pd.DataFrame({"Depth_m": f.Depth_m})
        feat["Change_1d_m"] = f.Depth_m.diff()
        feat["Mean_7d_m"] = f.Depth_m.rolling("7D", min_periods=3).mean()
        feat["Std_14d_m"] = f.Depth_m.rolling("14D", min_periods=5).std()
        feat["Mean_30d_m"] = f.Depth_m.rolling("30D", min_periods=10).mean()
        st.dataframe(rnd(feat.reset_index().tail(500)), hide_index=True, width="stretch", height=260)
        st.download_button("Download full station feature table", rnd(feat.reset_index(), 4).to_csv(index=False, date_format="%Y-%m-%d").encode(),
                           file_name="p17_station_features.csv", mime="text/csv")

# ====================================================================== Districts
with tabs[2]:
    ds_state = st.radio("State", ["Punjab", "Bihar"], horizontal=True, key="ds_state")
    D = dist[dist.State == ds_state].copy()
    st.markdown("##### District ranking")
    metric_opts = {"Latest_m": "Latest depth (m)", "Trend_m_per_yr": "Trend (m/year, + = falling)",
                   "Recharge_m": "Monsoon recharge (m, + = rose)", "Stations": "Stations"}
    pick = st.selectbox("Rank districts by", list(metric_opts), format_func=metric_opts.get)
    Dp = D.dropna(subset=[pick]).sort_values(pick, ascending=False)
    fig = px.bar(Dp, x="District", y=pick, color=pick, color_continuous_scale="RdYlBu_r" if pick != "Recharge_m" else "RdYlBu",
                 labels={pick: metric_opts[pick]}, height=380)
    st.plotly_chart(fig, width="stretch")
    if ds_state == "Punjab":
        st.caption("Trend: median of station changes between 2020 and 2026 over the same Mar–Oct days, per year. "
                   "Latest depth: each station's last 30 recorded days (Sep–Oct 2026).")
    else:
        st.caption("Trend: change in full-year median depth 2018 -> 2020, per year. Latest depth: Dec 2020.")
    # District-level relative watch label, computed only against districts in the selected state.
    depth_rank = D["Latest_m"].rank(pct=True)
    D["Relative watch"] = [
        L.relative_groundwater_status(p, t)
        for p, t in zip(depth_rank, D["Trend_m_per_yr"])
    ]
    st.markdown("##### District-level relative watch")
    st.dataframe(D[["District", "Latest_m", "Trend_m_per_yr", "Relative watch"]].rename(columns={
        "Latest_m": "Latest depth (m)", "Trend_m_per_yr": "Trend (m/year)"
    }).round(2), hide_index=True, width="stretch")
    st.caption("Relative comparison within the selected state only; not an official regulatory risk classification.")

    nice = D.rename(columns={"Latest_m": "Latest depth (m)", "Trend_m_per_yr": "Trend (m/yr)", "Falling_share": "Share of stations falling",
                             "Recharge_m": "Pre->post-monsoon recharge (m)", "Annual_rain_mm": "Annual normal rain (mm)",
                             "Monsoon_rain_mm": "Jun–Sep normal rain (mm)"}).drop(columns=["State", "Latest_date"])
    st.dataframe(rnd(nice, 2).sort_values("Latest depth (m)", ascending=False), hide_index=True, width="stretch", height=380)

    st.markdown("##### Seasonal pattern by district")
    chosen = st.multiselect("Districts to compare", sorted(D.District), default=sorted(D.District)[:3])
    if chosen:
        sc = seasonal[(seasonal.State == ds_state) & seasonal.District.isin(chosen)]
        fig = px.line(sc, x="Month", y="Anomaly_m", color="District", markers=True, height=360,
                      labels={"Anomaly_m": "Depth vs station's yearly mean (m)"})
        fig.update_yaxes(autorange="reversed")
        fig.update_xaxes(dtick=1)
        st.plotly_chart(fig, width="stretch")
        st.caption("Above the line = shallower than the station's own yearly average, below = deeper (axis flipped).")

# ====================================================================== Features & seasonality
with tabs[3]:
    fs = st.radio("State", ["Punjab", "Bihar"], horizontal=True, key="fs_state")
    cA, cB = st.columns(2)
    with cA:
        sm = seasonal[seasonal.State == fs].groupby("Month").apply(
            lambda g: np.average(g.Anomaly_m, weights=g.stations), include_groups=False).reset_index(name="Anomaly_m")
        fig = px.bar(sm, x="Month", y="Anomaly_m", title=f"{fs}: average seasonal swing in depth", height=340,
                     labels={"Anomaly_m": "Depth vs yearly mean (m)"}, color="Anomaly_m", color_continuous_scale="RdYlBu_r")
        fig.update_xaxes(dtick=1)
        st.plotly_chart(fig, width="stretch")
    with cB:
        rc = recharge[(recharge.State == fs)]
        rc = rc[rc.groupby("Year")["Station"].transform("size") >= 30]      # years with enough stations
        lo_, hi_ = rc["Recharge_m"].quantile([0.01, 0.99])
        rc = rc[rc["Recharge_m"].between(lo_, hi_)]                           # hide extreme sensor outliers
        fig = px.box(rc, x="Year", y="Recharge_m", points=False, height=340,
                     title=f"{fs}: pre- vs post-monsoon change per station",
                     labels={"Recharge_m": "Pre - post depth (m); + = water table rose"})
        fig.add_hline(y=0, line_dash="dot")
        st.plotly_chart(fig, width="stretch")
        st.caption("Pre-monsoon = 1 Apr–15 Jun median, post-monsoon = 1 Oct–15 Nov median. Punjab 2026 uses data to 26 Oct.")

    st.markdown("##### Does district rainfall explain depth?")
    DD = dist[dist.State == fs]
    ycol = st.selectbox("Compare annual normal rainfall with", ["Latest_m", "Trend_m_per_yr", "Recharge_m"],
                        format_func=lambda c: {"Latest_m": "Latest depth (m)", "Trend_m_per_yr": "Trend (m/yr)",
                                               "Recharge_m": "Monsoon recharge (m)"}[c])
    G = DD.dropna(subset=[ycol])
    fig = px.scatter(G, x="Annual_rain_mm", y=ycol, hover_name="District", height=380,
                     labels={"Annual_rain_mm": "District annual normal rainfall (mm)"})
    if len(G) > 3:
        k, b0 = np.polyfit(G["Annual_rain_mm"], G[ycol], 1)
        xs_ = np.array([G["Annual_rain_mm"].min(), G["Annual_rain_mm"].max()])
        fig.add_trace(go.Scatter(x=xs_, y=k * xs_ + b0, mode="lines", name="Linear fit", line=dict(dash="dash")))
    st.plotly_chart(fig, width="stretch")
    rr = rel.get(fs, {}).get(f"rain_vs_{ycol}")
    if rr:
        st.write(f"Spearman rank correlation **rho = {rr['rho']:+.2f}** (p = {rr['p']:.3f}, {rr['n_districts']} districts). "
                 "This is association between *district climate* and groundwater, not proof of cause.")

    ir = india_rain()
    if ir is not None:
        sub = ir[ir.SUBDIVISION == fs.upper()].sort_values("YEAR")
        sub = sub.assign(Rolling10=sub["ANNUAL"].rolling(10, min_periods=5).mean())
        fig = go.Figure()
        fig.add_trace(go.Bar(x=sub.YEAR, y=sub.ANNUAL, name="Annual rainfall (mm)", opacity=0.55))
        fig.add_trace(go.Scatter(x=sub.YEAR, y=sub.Rolling10, name="10-year mean", line=dict(width=3)))
        fig.update_layout(title=f"{fs}: annual rainfall 1901–2015 (supplied IMD subdivision data)", height=330)
        st.plotly_chart(fig, width="stretch")
        st.caption("This file ends in 2015, before the telemetry starts (2016), so it cannot be matched day-by-day to "
                   "groundwater. Rainfall enters the forecast model only as district climate normals.")

    st.markdown("##### Which inputs does the forecast model rely on?")
    h = st.select_slider("Forecast horizon (days)", options=L.HORIZONS, value=30)
    I = imps[(imps.State == fs) & (imps.Horizon_days == h)].copy()
    I["Feature"] = I["Feature"].map(lambda f: L.FEATURE_LABELS.get(f, f).replace("next h", f"next {h}"))
    I = I.sort_values("MAE_increase_m")
    fig = px.bar(I, x="MAE_increase_m", y="Feature", orientation="h", height=430,
                 labels={"MAE_increase_m": "Error increase when this input is shuffled (m)"})
    st.plotly_chart(fig, width="stretch")
    st.caption("Permutation importance measured on the hold-out period. Rainfall inputs are district *normals*, so their "
               "importance mostly reflects monsoon timing and district climate — not measured rainfall in a given year.")

# ====================================================================== Model performance
with tabs[4]:
    ms = st.radio("State", ["Punjab", "Bihar"], horizontal=True, key="mp_state")
    M = metrics[metrics.State == ms].copy()
    if meta:
        s_ = meta["summary"][ms]
        st.write(f"**Hold-out test:** train on data whose target date is before **{s_['cutoff_date']}**; test on origins "
                 f"from that date to **{s_['last_date']}** ({int(M['n_test'].min()):,}–{int(M['n_test'].max()):,} station forecasts per horizon, "
                 f"up to {int(M['test_stations'].max())} stations). The model never sees test-period data.")
    long = M.melt(id_vars="Horizon_days", value_vars=["MAE_model", "MAE_persistence", "MAE_seasonal_avg_change"],
                  var_name="Method", value_name="MAE (m)")
    long["Horizon"] = long["Horizon_days"].astype(str) + " d"
    long["Method"] = long["Method"].map({"MAE_model": "P17 model", "MAE_persistence": "Persistence (no change)",
                                         "MAE_seasonal_avg_change": "Seasonal-average change"})
    fig = px.bar(long, x="Horizon", y="MAE (m)", color="Method", barmode="group", height=360,
                 labels={"Horizon": "Forecast horizon"}, category_orders={"Horizon": [f"{h} d" for h in L.HORIZONS]})
    st.plotly_chart(fig, width="stretch")
    tbl = M[["Horizon_days", "MAE_model", "RMSE_model", "MAE_persistence", "Skill_vs_persistence_MAE",
             "Within_0.5m_model", "Within_0.5m_persistence"]].rename(columns={
        "Horizon_days": "Days ahead", "MAE_model": "Model MAE (m)", "RMSE_model": "Model RMSE (m)",
        "MAE_persistence": "Persistence MAE (m)", "Skill_vs_persistence_MAE": "Error reduction vs persistence",
        "Within_0.5m_model": "Within +/-0.5 m (model)", "Within_0.5m_persistence": "Within +/-0.5 m (persistence)"})
    st.dataframe(tbl.style.format({c: "{:.3f}" for c in tbl.columns if "(m)" in c}
                                  | {"Error reduction vs persistence": "{:.1%}", "Within +/-0.5 m (model)": "{:.1%}",
                                     "Within +/-0.5 m (persistence)": "{:.1%}"}), hide_index=True, width="stretch")
    hh = st.select_slider("Horizon for detail plots (days)", options=L.HORIZONS, value=30, key="mp_h")
    P = preds[(preds.State == ms) & (preds.Horizon_days == hh)]
    pa, pb = st.columns(2)
    with pa:
        samp = P.sample(min(3000, len(P)), random_state=0)
        fig = px.scatter(samp, x="Actual_m", y="Predicted_m", opacity=0.35, height=400, hover_name="Station",
                         title=f"Actual vs predicted depth, {hh} days ahead",
                         labels={"Actual_m": "Actual depth (m)", "Predicted_m": "Predicted depth (m)"})
        lim = [float(min(samp.Actual_m.min(), samp.Predicted_m.min())), float(max(samp.Actual_m.max(), samp.Predicted_m.max()))]
        fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines", name="Perfect", line=dict(dash="dash", color="grey")))
        st.plotly_chart(fig, width="stretch")
        r2 = 1 - ((P.Actual_m - P.Predicted_m) ** 2).sum() / ((P.Actual_m - P.Actual_m.mean()) ** 2).sum()
        r2p = 1 - ((P.Actual_m - P.Persistence_m) ** 2).sum() / ((P.Actual_m - P.Actual_m.mean()) ** 2).sum()
        st.caption(f"R^2 on depth level: model {r2:.3f}, persistence {r2p:.3f}. Level R^2 is high for any method because "
                   "stations differ a lot in depth; the error comparison on the left is the fairer test.")
    with pb:
        BM = by_month[(by_month.State == ms) & (by_month.Horizon_days == hh)].melt(
            id_vars="Month", value_vars=["MAE_model", "MAE_persistence"], var_name="Method", value_name="MAE (m)")
        BM["Method"] = BM["Method"].map({"MAE_model": "P17 model", "MAE_persistence": "Persistence"})
        fig = px.bar(BM, x="Month", y="MAE (m)", color="Method", barmode="group", height=400,
                     title="Error by month of forecast origin")
        fig.update_xaxes(dtick=1)
        st.plotly_chart(fig, width="stretch")
    BD = by_dist[(by_dist.State == ms) & (by_dist.Horizon_days == hh)].sort_values("MAE_model")
    fig = px.bar(BD, x="District", y=["MAE_model", "MAE_persistence"], barmode="group", height=360,
                 title=f"Hold-out error by district, {hh} days ahead", labels={"value": "MAE (m)", "variable": ""})
    st.plotly_chart(fig, width="stretch")
    st.info("Read this honestly: the model beats the simple baselines on the hold-out, but groundwater depends on pumping and "
            "rainfall that are not in the inputs. Errors grow with horizon, and a single hold-out season is not a guarantee for "
            "other seasons or years.")

# ====================================================================== Data quality
with tabs[5]:
    st.markdown("##### What was done to the raw telemetry")
    fl = pd.DataFrame(qrep.get("files", []))
    if not fl.empty:
        st.dataframe(fl.rename(columns={
            "file": "Source file", "rows_raw": "Raw rows", "dropped_unparseable_time_or_value": "Unreadable time/value",
            "negative_values_share": "Share of negative values", "dropped_placeholder_ge_150m": "Dropped: |value| >= 150 m",
            "dropped_exact_zero": "Dropped: exact 0", "rows_kept": "Valid readings"}), hide_index=True, width="stretch")
    sp = qrep.get("spike_filter", {})
    st.write(f"Spike filter: daily values more than **{sp.get('threshold_m')} m** from their centred "
             f"{sp.get('window')} median were removed (**{sp.get('station_days_removed', 0):,}** of "
             f"{sp.get('station_days_before', 0):,} station-days). Stray station-years with under 20 observed days were dropped.")
    st.markdown(
        "- **Depth convention:** Bihar files store positive depths. Punjab files mix negative and positive values between "
        "stations, so the magnitude is used as depth below ground. Values >=150 m (e.g. 999, 958) are placeholders/elevations.\n"
        "- **Flagged stations** (not forecast): daily std > 15 m, median depth >= 90 m, or a sensor that never changes.\n"
        "- **No interpolation:** days without a valid reading are missing, not filled.\n"
        "- **The file labelled 2026–2030** actually contains real readings from 1 Jan to 26 Oct 2026 and is used as such; "
        "nothing after 26 Oct exists in it.")
    sus = stations[stations.suspect][["State", "District", "StationLabel", "obs_days", "median_depth", "depth_std"]]
    st.markdown(f"##### Flagged stations ({len(sus)})")
    st.dataframe(rnd(sus, 2), hide_index=True, width="stretch", height=260)
    fig = px.histogram(stations[stations.usable], x="coverage", color="State", nbins=40, height=300,
                       title="Daily coverage per station (within the years it reports)")
    st.plotly_chart(fig, width="stretch")

# ====================================================================== Export
with tabs[6]:
    st.markdown("##### Downloads")
    e1, e2 = st.columns(2)
    e1.download_button("District summary (CSV)", rnd(dist).to_csv(index=False, date_format="%Y-%m-%d").encode(), "p17_district_summary.csv", "text/csv")
    e1.download_button("Station summary (CSV)",
                       rnd(sumry).to_csv(index=False, date_format="%Y-%m-%d").encode(),
                       "p17_station_summary.csv", "text/csv")
    e2.download_button("Back-test metrics (CSV)", rnd(metrics, 4).to_csv(index=False).encode(), "p17_backtest_metrics.csv", "text/csv")
    e2.download_button("Feature importance (CSV)", rnd(imps, 5).to_csv(index=False).encode(), "p17_feature_importance.csv", "text/csv")
    st.markdown("##### Full daily records for one state")
    ex_state = st.selectbox("State", ["Punjab", "Bihar"], key="ex_state")
    n_ex = int((daily.State == ex_state).sum())
    st.download_button(f"Download all {ex_state} daily records ({n_ex:,} rows)", state_csv(ex_state),
                       f"p17_{ex_state.lower()}_daily.csv", "text/csv")

st.divider()
st.caption("P17 · SIH1696 · Source: supplied Punjab/Bihar groundwater telemetry and IMD rainfall normals. "
           "Forecasts are statistical estimates, not official measurements.")


# ====================================================================== Sources & deployment
with tabs[7]:
    st.markdown("### Dataset provenance and coverage")
    st.warning("All-India coverage is the project target, not the current verified dataset. The bundled trained models and processed observations cover Punjab and Bihar only. Do not interpret the dashboard as all-India until additional official state datasets are ingested and validated.")
    fin = qrep.get("final", {})
    st.write({
        "Verified states in current data": fin.get("states", []),
        "Districts": fin.get("districts"),
        "Stations": fin.get("stations"),
        "Daily station records": fin.get("station_days"),
        "First observation": fin.get("first_date"),
        "Last observation": fin.get("last_date"),
        "Data refresh mode": "Bundled dataset snapshot; source API ingestion is not yet configured",
    })
    st.markdown("#### Official source links")
    st.markdown("- [National Water Data Portal (NWIC)](https://nwdp.nwic.gov.in/) — discover groundwater datasets and downloads.")
    st.markdown("- [CGWB groundwater-level monitoring](https://cgwb.gov.in/hi/ground-water-level-monitoring) — official monitoring page.")
    st.markdown("- [Quarterly manual groundwater dataset](https://nwdp.nwic.gov.in/dataset/gwl-manual-quarterly-central-ground-water-board-department).")
    st.markdown("- [Telemetry groundwater dataset](https://nwdp.nwic.gov.in/dataset/ground-water-level-telemetry-hourly-central-ground-water-board-cgwb).")
    st.markdown("- [IMD historical weather API](https://open-meteo.com/en/docs/historical-weather-api) — supplementary weather features, not a replacement for groundwater observations.")
    st.markdown("#### Database and data refresh status")
    st.info("The dashboard's existing charts still use the verified bundled Punjab/Bihar snapshot. A database can store newly ingested official observations and sync history; it does not create live groundwater readings unless the official source publishes new observations.")
    try:
        from database_store import database_status, sync_normalized_csv
        db_status = database_status()
        st.write({"Database mode": db_status["mode"], "Database configured": db_status["configured"],
                  "Observation rows stored": db_status["observation_rows"],
                  "Last successful sync (UTC)": db_status["last_sync_utc"] or "Never"})
        normalized = L.ROOT / "data" / "processed" / "official" / "normalized_observations.csv"
        if normalized.exists():
            st.caption(f"Found normalized official staging file: {normalized.name} ({normalized.stat().st_size:,} bytes).")
            if st.button("Sync normalized observations to database", type="primary"):
                with st.spinner("Syncing observations in chunks; large files may take a while..."):
                    result = sync_normalized_csv(normalized)
                st.success(f"Database sync finished: {result['rows_written']:,} rows written; {result['rows_skipped']:,} invalid rows skipped.")
                st.rerun()
        else:
            st.warning("No normalized all-India official file found yet. Download and normalize official resources first; see README.md for commands.")
    except Exception as exc:
        st.warning(f"Database status unavailable: {exc}")
    if st.button("Refresh dashboard snapshot", help="Clears cached bundled data; this is not a source download."):
        st.cache_data.clear()
        st.rerun()
    st.markdown("#### Enable login on deployment")
    st.write("Set `SUPABASE_URL` and `SUPABASE_ANON_KEY` in Streamlit Cloud -> App settings -> Secrets. Login, account creation and password reset are enabled when both are configured. Without them, the app explicitly runs in demo mode.")
    st.markdown("#### Deployment checklist")
    st.markdown("""- [ ] Push this project to a private or public GitHub repository.
- [ ] Deploy `app.py` on Streamlit Community Cloud.
- [ ] Configure Supabase Auth and allowed redirect URLs before enabling user login.
- [ ] Add `SUPABASE_URL` and `SUPABASE_ANON_KEY` as deployment secrets.
- [ ] Run the smoke test in the deployment environment.
- [ ] Configure a verified groundwater source endpoint and schedule before claiming live data.""")

