import numpy as np, pandas as pd, plotly.express as px, streamlit as st, ui
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
raw, X, y, A = ui.load(); d = raw.assign(rev=y.values, age=X.days_open.values / 365)
ui.hero("Module 03 · Evidence", "Analytics studio", "Every chart behind the model, interactive and exportable.", decor=False)
t = st.tabs(["Revenue", "Cities & types", "Maturity", "Correlations", "PCA map", "Drivers"])
with t[0]:
    st.plotly_chart(ui.style(px.histogram(d, x="rev", nbins=30, marginal="box", title="Revenue distribution (right-skewed)"), 420), use_container_width=True)
with t[1]:
    a, b = st.columns(2)
    top = d.groupby("City").rev.agg(["median", "count"]).sort_values("count", ascending=False).head(10).reset_index()
    a.plotly_chart(ui.style(px.bar(top, x="City", y="median", color="count", color_continuous_scale=["#2a1d17", ui.CLAY], title="Median revenue, 10 most common cities")), use_container_width=True)
    b.plotly_chart(ui.style(px.violin(d, x="Type", y="rev", color="Type", box=True, title="Revenue by type")), use_container_width=True)
    st.plotly_chart(ui.style(px.sunburst(d, path=["City Group", "Type"], values="rev", color_discrete_sequence=[ui.CLAY, ui.GOLD, ui.MOSS], title="Revenue share"), 420), use_container_width=True)
with t[2]:
    st.plotly_chart(ui.style(px.scatter(d, x="age", y="rev", color="City Group", size="P29", trendline="lowess", title="Restaurant age vs revenue"), 450), use_container_width=True)
with t[3]:
    k = st.slider("Top N features by |correlation| with revenue", 8, 40, 14); c = X.assign(revenue=y).corr()
    cols = c["revenue"].abs().sort_values(ascending=False).index[:k]
    st.plotly_chart(ui.style(px.imshow(c.loc[cols, cols], color_continuous_scale="RdBu_r", zmin=-1, zmax=1, text_auto=".2f"), 600), use_container_width=True)
with t[4]:
    Z = PCA(3).fit_transform(StandardScaler().fit_transform(X)); pc = pd.DataFrame(Z, columns=["PC1", "PC2", "PC3"]).assign(rev=y.values / 1e6, group=d["City Group"].values)
    st.plotly_chart(ui.style(px.scatter_3d(pc, x="PC1", y="PC2", z="PC3", color="rev", symbol="group", color_continuous_scale=["#4F6B4A", "#C9A24B", "#B4492D"], title="Branches in PCA space"), 560), use_container_width=True)
    ev = np.cumsum(A["pca_var"]); st.plotly_chart(ui.style(px.area(x=range(1, len(ev) + 1), y=ev, labels={"x": "Components", "y": "Cumulative variance"}, title=f"{int((ev<.95).sum()+1)} components keep 95% variance")), use_container_width=True)
with t[5]:
    fi = A["importance"].head(12).sort_values(); st.plotly_chart(ui.style(px.bar(x=fi.values, y=fi.index, orientation="h", title="Random Forest feature importance"), 420), use_container_width=True)
    g = d.groupby("City Group").rev.median(); st.markdown(f"- Big-City median **${g.get('Big Cities',0)/1e6:.2f}M** vs other **${g.get('Other',0)/1e6:.2f}M**\n- Strongest indicator: **{A['importance'].index[0]}**\n- Age/revenue correlation: **{np.corrcoef(X.days_open, y)[0,1]:+.2f}**")
