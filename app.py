import streamlit as st, ui, pipeline as pl
st.set_page_config("RevenuIQ", "◐", layout="wide", initial_sidebar_state="collapsed")
ui.css()
with st.sidebar:
    st.markdown("### Data")
    up = st.file_uploader("Replace training data (train.csv)", type="csv")
    if up:
        (pl.DATA.parent / ".synthetic").unlink(missing_ok=True); pl.DATA.write_bytes(up.getvalue()); pl.MODEL.unlink(missing_ok=True); st.cache_resource.clear(); st.rerun()
    if st.button("Retrain models"): pl.MODEL.unlink(missing_ok=True); st.cache_resource.clear(); st.rerun()
if pl.load_raw() is None: st.error("Put Kaggle's train.csv in data/ first."); st.stop()
if (pl.DATA.parent / ".synthetic").exists(): st.warning("Running on SYNTHETIC demo data. Upload the real Kaggle TFI train.csv in the sidebar.")
st.navigation([st.Page("views/home.py", title="Home", icon="🏠", default=True), st.Page("views/evaluate.py", title="Site Evaluator", icon="📍"),
    st.Page("views/compare.py", title="Compare & Shortlist", icon="⚖️"), st.Page("views/footfall.py", title="Attendance Forecast", icon="👥"), st.Page("views/analytics.py", title="Analytics", icon="📊"),
    st.Page("views/models.py", title="Model Lab", icon="🧪")], position="top").run()
