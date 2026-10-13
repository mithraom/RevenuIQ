import numpy as np, pandas as pd, plotly.express as px, plotly.graph_objects as go, streamlit as st, ui, attendance as at
ui.hero("Module 02 · Attendance", "How many guests will walk in?", "Daily visitor forecasts for each restaurant, up to a week ahead, trained on real visit counts and validated on future weeks, never shuffled data.", decor=False)
if not at.available():
    st.error("Put air_visit_data.csv, air_store_info.csv, date_info.csv (and air_reserve.csv) from Kaggle's Recruit Restaurant Visitor Forecasting in data/attendance/."); st.stop()
A = ui.load_att(); fc, hist, bt, stores, res = A["forecast"], A["history"], A["backtest"], A["stores"], A["results"]
DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]; band = np.exp(A["rmsle"])
t1, t2, t3 = st.tabs(["Store forecast", "Network insights", "Model lab"])
with t1:
    n = hist.groupby("air_store_id").size(); ok = fc.air_store_id.unique(); ok = [s for s in ok if n.get(s, 0) >= 60]
    info = stores.set_index("air_store_id"); lab = {s: f"{s[-6:]} · {info.loc[s, 'air_genre_name']} · {info.loc[s, 'prefecture']}" for s in ok}
    sid = st.selectbox("Restaurant", ok, format_func=lambda s: lab[s])
    h, f, b = hist[hist.air_store_id == sid].tail(70), fc[fc.air_store_id == sid], bt[bt.air_store_id == sid]
    last7 = hist[(hist.air_store_id == sid) & (hist.visit_date > A["last"] - pd.Timedelta(days=7))].visitors.sum()
    c = st.columns(3); ui.card(c[0], f"{f.forecast.sum():.0f}", "Guests forecast, next 7 days"); ui.card(c[1], f"{f.loc[f.forecast.idxmax(), 'forecast']:.0f}", f"Busiest day · {DOW[int(f.loc[f.forecast.idxmax(), 'dow'])]}")
    ui.card(c[2], f"{(f.forecast.sum()-last7)/max(last7,1)*100:+.0f}%", "vs the last 7 recorded days")
    g = go.Figure(); g.add_scatter(x=h.visit_date, y=h.visitors, name="Actual", line=dict(color=ui.INK, width=2))
    if len(b): g.add_scatter(x=b.visit_date, y=b.pred, name="Backtest prediction", line=dict(color=ui.GOLD, dash="dot"))
    g.add_scatter(x=f.visit_date, y=f.forecast * band, line=dict(width=0), showlegend=False); g.add_scatter(x=f.visit_date, y=f.forecast / band, fill="tonexty", fillcolor="rgba(224,103,63,.2)", line=dict(width=0), name="Likely range")
    g.add_scatter(x=f.visit_date, y=f.forecast, name="Forecast", line=dict(color=ui.CLAY, width=3)); g.update_layout(title="Last 10 weeks, backtest and 7-day forecast", yaxis_title="Visitors")
    st.plotly_chart(ui.style(g, 420), use_container_width=True)
    tb = f.assign(Day=f.dow.map(lambda i: DOW[int(i)]), Date=f.visit_date.dt.date, Holiday=f.holiday_flg.map({1: "Yes", 0: "No"}), Booked_7d_ahead=f.reserved_7d, Forecast=f.forecast.round(0))
    st.dataframe(tb[["Date", "Day", "Holiday", "Booked_7d_ahead", "Forecast"]], hide_index=True, use_container_width=True)
    st.markdown(f'<div class="note">Typical error is about a factor of {band:.2f} (RMSLE {A["rmsle"]:.2f}). The forecast uses only information available 7 days earlier: past visits, holidays and reservations booked at least a week ahead.</div>', unsafe_allow_html=True)
with t2:
    d = hist.merge(stores, on="air_store_id"); tot = d.groupby("visit_date").visitors.sum().reset_index()
    st.plotly_chart(ui.style(px.line(tot, x="visit_date", y="visitors", title="Total daily visitors, all restaurants"), 340), use_container_width=True)
    a, b = st.columns(2); d["Day"] = d.visit_date.dt.dayofweek.map(lambda i: DOW[i]); dw = d.groupby("Day").visitors.mean().reindex(DOW)
    a.plotly_chart(ui.style(px.bar(dw, title="Average visitors by weekday", labels={"value": "Visitors", "Day": ""}), 320), use_container_width=True)
    hol = A["holidays"].set_index("visit_date").holiday_flg; d["Holiday"] = d.visit_date.map(hol).map({1: "Holiday", 0: "Normal day"})
    b.plotly_chart(ui.style(px.bar(d.groupby("Holiday").visitors.mean(), title="Holiday effect (average visitors per restaurant)", labels={"value": "Visitors"}), 320), use_container_width=True)
    a.plotly_chart(ui.style(px.bar(d.groupby("air_genre_name").visitors.mean().sort_values(), orientation="h", title="Average visitors by cuisine", labels={"value": "Visitors", "air_genre_name": ""}), 420), use_container_width=True)
    b.plotly_chart(ui.style(px.bar(d.groupby("prefecture").visitors.mean().sort_values(), orientation="h", title="Average visitors by region", labels={"value": "Visitors", "prefecture": ""}), 420), use_container_width=True)
with t3:
    st.markdown("Validated on **two future 4-week windows** (train on the past, test on what follows). RMSLE is the competition metric; lower is better.")
    st.dataframe(res.round(3), hide_index=True, use_container_width=True)
    f2 = go.Figure(go.Bar(y=res.Model, x=res.RMSLE, orientation="h", marker_color=[ui.CLAY] + [ui.SAND] * (len(res) - 1))); f2.update_layout(title="RMSLE by model", yaxis=dict(autorange="reversed"))
    a, b = st.columns(2); a.plotly_chart(ui.style(f2, 380), use_container_width=True)
    imp = A["importance"].head(10)[::-1]; b.plotly_chart(ui.style(px.bar(x=imp.values, y=imp.index, orientation="h", title="What drives attendance (permutation importance)", labels={"x": "Error increase when shuffled", "y": ""}), 380), use_container_width=True)
    if A["tuning"]:
        t = A["tuning"]; st.markdown(f"**Hypertuning:** {t['model']}, best settings `{t['params']}`. RMSLE {t['before']['RMSLE']:.3f} → {t['after']['RMSLE']:.3f}.")
    s = bt.sample(min(4000, len(bt)), random_state=1); m = max(s.visitors.max(), s.pred.max())
    p = px.scatter(s, x="visitors", y="pred", opacity=.4, labels={"visitors": "Actual", "pred": "Predicted"}, title="Last 4 weeks: predicted vs actual"); p.add_shape(type="line", x0=0, y0=0, x1=m, y1=m, line=dict(color=ui.GOLD, dash="dash"))
    c, d2 = st.columns(2); c.plotly_chart(ui.style(p, 360), use_container_width=True)
    e = bt.assign(err=(bt.pred - bt.visitors).abs()).groupby("dow").err.mean(); d2.plotly_chart(ui.style(px.bar(x=[DOW[i] for i in e.index], y=e.values, title="Mean absolute error by weekday", labels={"x": "", "y": "Visitors"}), 360), use_container_width=True)
