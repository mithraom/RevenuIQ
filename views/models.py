import plotly.express as px, plotly.graph_objects as go, streamlit as st, ui, pipeline as pl
from sklearn.model_selection import cross_val_predict
raw, X, y, A = ui.load(); res = A["results"]
ui.hero("Module 04", "Model lab", "Nine models, 5×3 repeated cross-validation, no leakage: PCA and correlation filtering are fitted inside each fold.", decor=False)
s = res.assign(**{"RMSE ($M)": res.RMSE / 1e6, "MAE ($M)": res.MAE / 1e6})[["Model", "RMSE ($M)", "MAE ($M)", "R2"]].round(3)
st.dataframe(s, hide_index=True, use_container_width=True)
a, b = st.columns(2)
f = go.Figure(go.Bar(y=res.Model, x=res.RMSE / 1e6, orientation="h", error_x=dict(array=res.RMSE_sd / 1e6), marker_color=[ui.CLAY] + [ui.SAND] * (len(res) - 1)))
f.update_layout(title="CV RMSE ($M), lower is better", yaxis=dict(autorange="reversed")); a.plotly_chart(ui.style(f, 360), use_container_width=True)
pr = cross_val_predict(pl.get_model(A["best"]), X, y, cv=5)
g = px.scatter(x=y / 1e6, y=pr / 1e6, labels={"x": "Actual ($M)", "y": "Predicted ($M)"}, title=f"Out-of-fold predictions · {A['best']}")
g.add_shape(type="line", x0=0, y0=0, x1=y.max() / 1e6, y1=y.max() / 1e6, line=dict(color=ui.GOLD, dash="dash")); b.plotly_chart(ui.style(g, 360), use_container_width=True)
st.markdown(f'<div class="note">Selected: <b>{A["best"]}</b>. With few rows and noisy predictors, modest R² is expected; report it honestly and discuss the overfitting risk.</div>', unsafe_allow_html=True)
