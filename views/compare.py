import numpy as np, pandas as pd, plotly.graph_objects as go, plotly.express as px, streamlit as st, ui, pipeline as pl
raw, X, y, A = ui.load(); imp = A["importance"]; band = A["results"].loc[0, "RMSE"]
ui.hero("Module 02 · Shortlisting", "Which site wins?", "Put two candidates head to head, or upload a whole shortlist and rank it.", decor=False)
t2, t3 = st.tabs(["Compare A vs B", "Batch CSV"])
with t2:
    a, b = st.columns(2, gap="large")
    with a: st.markdown("##### Site A"); ra = ui.site_inputs("a", X, imp, A)
    with b: st.markdown("##### Site B"); rb = ui.site_inputs("b", X, imp, A)
    pa, pb = ui.predict(A, [ra, rb]); w = "A" if pa >= pb else "B"
    m = st.columns(3); m[0].metric("Site A", f"${pa/1e6:.2f}M"); m[1].metric("Site B", f"${pb/1e6:.2f}M"); m[2].metric("Better site", w, f"{abs(pa-pb)/1e6:.2f}M gap")
    st.plotly_chart(ui.style(px.bar(x=["Site A", "Site B"], y=[pa / 1e6, pb / 1e6], color=["A", "B"], color_discrete_sequence=[ui.CLAY, ui.GOLD], labels={"x": "", "y": "Revenue ($M)"}), 300), use_container_width=True)
with t3:
    st.caption("Upload a CSV with the same columns as train.csv (revenue optional). Download scored results.")
    st.download_button("Download template", raw.drop(columns="revenue").head(5).to_csv(index=False), "template.csv")
    up = st.file_uploader("Candidate sites", type="csv", key="batch")
    if up:
        d = pd.read_csv(up); Xb, _ = pl.preprocess(d); d["predicted_revenue"] = ui.predict(A, Xb).round(0)
        st.dataframe(d, use_container_width=True); st.download_button("Download predictions", d.to_csv(index=False), "predictions.csv")
        st.plotly_chart(ui.style(px.bar(d.sort_values("predicted_revenue"), x="predicted_revenue", y=d.index.astype(str) if "Id" not in d else "Id", orientation="h", title="Ranked shortlist")), use_container_width=True)
