import numpy as np, pandas as pd, plotly.graph_objects as go, streamlit as st, ui
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
raw, X, y, A = ui.load(); imp = A["importance"]; band = A["results"].loc[0, "RMSE"]
ui.hero("Module 01 · Site evaluation", "Should we open here?", "Describe a site. Get a revenue forecast, a verdict, comparable branches and the levers that move the number.")
l, r = st.columns([1, 1.45], gap="large")
with l:
    st.markdown("##### Site profile"); row = ui.site_inputs("s", X, imp, A)
p = float(ui.predict(A, [row])[0]); pct = (y < p).mean() * 100
verdict, vc = ("OPEN", "#5FA37A") if pct >= 66 else ("REVIEW", "#D9B45A") if pct >= 33 else ("AVOID", "#E0673F")
tops = [i for i in imp.index if i.startswith("P")][:4]; cols = tops + ["days_open", "is_big_city"]
sc = StandardScaler().fit(X[cols]); nn = NearestNeighbors(n_neighbors=5).fit(sc.transform(X[cols]))
idx = nn.kneighbors(sc.transform(pd.DataFrame([row])[cols]))[1][0]
peers = raw.iloc[idx][["Id", "City", "Type"]].assign(Revenue=y.iloc[idx].values.round(0), Years_open=(X.days_open.iloc[idx] / 365).round(1).values)
pm = peers.Revenue.mean()
with r:
    st.markdown(f'<div class="gc"><span class="l">FORECAST ANNUAL REVENUE</span><div class="big">${p/1e6:.2f}M</div>'
                f'<span class="l">likely range ${max(p-band,0)/1e6:.2f}M – ${(p+band)/1e6:.2f}M &nbsp;·&nbsp; beats {pct:.0f}% of branches</span><br><br>'
                f'<span class="pill" style="color:{vc}">VERDICT: {verdict}</span></div>', unsafe_allow_html=True)
    g = go.Figure(go.Indicator(mode="gauge+number", value=pct, number={"suffix": "th percentile"}, gauge={"axis": {"range": [0, 100]}, "bar": {"color": vc},
        "bgcolor": "rgba(255,255,255,.04)", "steps": [{"range": [0, 33], "color": "#2a1d17"}, {"range": [33, 66], "color": "#33281a"}, {"range": [66, 100], "color": "#1f2d24"}]}))
    st.plotly_chart(ui.style(g, 230), use_container_width=True)
a, b = st.columns(2, gap="large")
with a:
    st.markdown("##### Comparable branches"); st.dataframe(peers, hide_index=True, use_container_width=True)
    st.markdown(f'<div class="note">Peer average <b>${pm/1e6:.2f}M</b> vs forecast <b>${p/1e6:.2f}M</b> ({(p-pm)/pm*100:+.0f}%). A large gap means the model and its nearest real branches disagree: investigate before deciding.</div>', unsafe_allow_html=True)
with b:
    d = []
    for k in tops:
        lo, hi = float(ui.predict(A, [dict(row, **{k: X[k].quantile(.1)})])[0]), float(ui.predict(A, [dict(row, **{k: X[k].quantile(.9)})])[0]); d.append((k, lo - p, hi - p))
    t = pd.DataFrame(d, columns=["Lever", "Low (10th pct)", "High (90th pct)"])
    f = go.Figure([go.Bar(y=t.Lever, x=t[c] / 1e6, name=c, orientation="h", marker_color=m) for c, m in [("Low (10th pct)", ui.CLAY), ("High (90th pct)", ui.MOSS)]])
    f.update_layout(barmode="relative", title="Lever sensitivity ($M change)", xaxis_title="Revenue change ($M)"); st.plotly_chart(ui.style(f, 300), use_container_width=True)
days = np.arange(0, 8001, 250); c = ui.predict(A, [dict(row, days_open=x) for x in days])
f = go.Figure(go.Scatter(x=days, y=c / 1e6, fill="tozeroy", line=dict(color=ui.GOLD), fillcolor="rgba(217,180,90,.15)")); f.add_vline(x=row["days_open"], line_color=ui.CLAY, line_width=3)
f.update_layout(title="Maturity curve: revenue as the branch ages", xaxis_title="Days since opening", yaxis_title="Revenue ($M)"); st.plotly_chart(ui.style(f, 280), use_container_width=True)
rep = f"# RevenuIQ site report\n\n- Forecast: ${p/1e6:.2f}M (range ${max(p-band,0)/1e6:.2f}M-${(p+band)/1e6:.2f}M)\n- Percentile vs existing branches: {pct:.0f}\n- Verdict: {verdict}\n- Peer average: ${pm/1e6:.2f}M\n- Model: {A['best']}\n\n## Inputs\n" + "\n".join(f"- {k}: {row[k]:.2f}" for k in cols + ["type_FC", "type_IL", "type_DT", "type_MB"])
st.download_button("Download site report", rep, "site_report.md")
