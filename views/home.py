import streamlit as st, ui
raw, X, y, A = ui.load(); res = A["results"]
ui.hero("RevenuIQ · site-evaluation decision support", "Know a site's worth before you sign.", "Two modules: a cross-validated revenue model that turns a site into an open / review / avoid decision, and a daily attendance forecaster built on real visit counts.")
c = st.columns(4)
ui.card(c[0], len(raw), "Branches analysed"); ui.card(c[1], f"${y.median()/1e6:.2f}M", "Median revenue")
ui.card(c[2], f"{res.loc[0,'R2']:.2f}", f"CV R² · {A['best']}"); ui.card(c[3], f"${res.loc[0,'RMSE']/1e6:.2f}M", "CV RMSE")
st.write(""); st.subheader("The problem")
st.markdown("Opening a restaurant in the wrong place is expensive. Past work stops at model accuracy on one dataset and rarely handles 37 correlated predictors on only 137 rows. RevenuIQ closes three gaps:")
f = st.columns(3)
for col, t, d in [(f[0], "01 · Tame the dimensions", "Correlation filtering and PCA inside every cross-validation fold, so 37 correlated predictors don't overfit 137 rows."),
                  (f[1], "02 · Compare honestly", "Linear Regression, Random Forest and SVR, with and without PCA, scored by repeated 5-fold CV."),
                  (f[2], "03 · Decide, not just predict", "Forecast, percentile, comparable branches and lever sensitivity become one verdict per site.")]:
    col.markdown(f'<div class="gc"><span class="l">{t}</span><p>{d}</p></div>', unsafe_allow_html=True)
st.write(""); n = st.columns(5)
for col, p, lab in zip(n, ["views/evaluate.py", "views/compare.py", "views/footfall.py", "views/analytics.py", "views/models.py"], ["Evaluate a site →", "Compare & shortlist →", "Attendance forecast →", "Open analytics →", "Open model lab →"]): col.page_link(p, label=lab)
