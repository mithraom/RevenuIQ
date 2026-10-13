import numpy as np, pandas as pd, streamlit as st
import pipeline as pl
INK, CLAY, GOLD, MOSS, SAND, CREAM = "#F3EBDD", "#E0673F", "#D9B45A", "#5FA37A", "#3A3028", "#F6F2EA"

LABELS = {  # ILLUSTRATIVE labels for anonymised TFI indicators (not the real definitions)
    "P1": "Population density index",
    "P2": "Foot traffic index",
    "P6": "Parking & access score",
    "P8": "Mall / retail anchor score",
    "P11": "Competitor presence index",
    "P13": "Public transport proximity",
    "P21": "Office & retail density",
    "P22": "Neighbourhood growth index",
    "P28": "Household income index",
    "P29": "Residential area size index",
}

def lab(k):
    return f"{LABELS[k]} ({k})" if k in LABELS else k

def svg(kind, size, style):
    s = {"plate": '<circle cx="50" cy="50" r="46" fill="#F6F2EA"/><circle cx="50" cy="50" r="34" fill="none" stroke="#C9A24B" stroke-width="2"/><circle cx="50" cy="50" r="22" fill="#B4492D" opacity=".85"/><path d="M42 44q8-10 16 0" stroke="#F6F2EA" stroke-width="3" fill="none"/>',
         "coin": '<circle cx="50" cy="50" r="44" fill="#C9A24B"/><circle cx="50" cy="50" r="34" fill="none" stroke="#F6F2EA" stroke-width="2" opacity=".8"/><text x="50" y="62" font-size="38" text-anchor="middle" fill="#1F1B16" font-family="serif">$</text>',
         "leaf": '<path d="M10 90C10 40 45 10 90 10 90 55 60 90 10 90Z" fill="#4F6B4A"/><path d="M10 90L70 30" stroke="#F6F2EA" stroke-width="3"/>',
         "cup": '<path d="M18 30h56v26a28 28 0 0 1-56 0Z" fill="#F6F2EA"/><path d="M74 36h8a10 10 0 0 1 0 20h-8" stroke="#F6F2EA" stroke-width="5" fill="none"/><path d="M30 14q6 8 0 14M46 14q6 8 0 14" stroke="#C9A24B" stroke-width="3" fill="none"/>'}[kind]
    return f'<svg class="fl" viewBox="0 0 100 100" width="{size}" style="{style}">{s}</svg>'

def css():
    st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600;9..144,700&family=DM+Sans:wght@400;500;700&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif}
.stApp{background:#0E0B09;color:#F3EBDD}
.stApp::before{content:"";position:fixed;inset:-20%;z-index:0;pointer-events:none;background:radial-gradient(600px 420px at 15% 10%,rgba(224,103,63,.30),transparent 60%),radial-gradient(700px 500px at 90% 20%,rgba(217,180,90,.20),transparent 60%),radial-gradient(650px 500px at 50% 105%,rgba(95,163,122,.22),transparent 60%);animation:drift 16s ease-in-out infinite alternate}
@keyframes drift{to{transform:translate(3%,4%) scale(1.12)}}
@keyframes fl{0%,100%{transform:translateY(0) rotate(0)}50%{transform:translateY(-20px) rotate(7deg)}}
@keyframes rise{from{opacity:0;transform:translateY(16px)}to{opacity:1;transform:none}}
@keyframes sheen{to{background-position:200% center}}
.block-container{position:relative;z-index:1}
h1,h2,h3,h4{font-family:'Fraunces',serif!important;letter-spacing:-.02em;color:#F3EBDD}
header[data-testid="stHeader"]{background:transparent}#MainMenu,footer{visibility:hidden}
.fl{position:absolute;animation:fl 7s ease-in-out infinite;filter:drop-shadow(0 18px 24px rgba(0,0,0,.6))}
.hero2{position:relative;overflow:hidden;border-radius:26px;padding:3.2rem;margin:.6rem 0 1.6rem;border:1px solid rgba(255,255,255,.12);
 background:linear-gradient(135deg,rgba(255,255,255,.08),rgba(255,255,255,.02));backdrop-filter:blur(14px);animation:rise .8s both;box-shadow:0 30px 70px -30px rgba(224,103,63,.55)}
.hero2 .k{letter-spacing:.28em;text-transform:uppercase;font-size:.7rem;color:#D9B45A}
.hero2 h1{font-size:3.9rem;line-height:1;margin:.5rem 0 .8rem;max-width:660px;background:linear-gradient(90deg,#F3EBDD,#D9B45A,#E0673F,#F3EBDD);background-size:200%;-webkit-background-clip:text;color:transparent;animation:sheen 8s linear infinite}
.hero2 p{max-width:540px;color:#b9ad9b;font-size:1.02rem;position:relative;z-index:2}
.gc{background:rgba(255,255,255,.05);backdrop-filter:blur(12px);border:1px solid rgba(255,255,255,.10);border-radius:20px;padding:1.2rem 1.4rem;
 box-shadow:0 20px 40px -24px rgba(0,0,0,.8);transition:.3s;animation:rise .8s both;height:100%}
.gc:hover{transform:translateY(-6px);border-color:rgba(217,180,90,.5);box-shadow:0 26px 50px -20px rgba(224,103,63,.45)}
.gc b.n{font-family:'Fraunces',serif;font-size:2.2rem;display:block;color:#F3EBDD}
.gc span.l{font-size:.7rem;text-transform:uppercase;letter-spacing:.14em;color:#9a8f7e}
.gc p{color:#b9ad9b;margin:.3rem 0 0}
.big{font-family:'Fraunces',serif;font-size:4.6rem;font-weight:700;line-height:1.05;background:linear-gradient(90deg,#D9B45A,#E0673F);-webkit-background-clip:text;color:transparent;filter:drop-shadow(0 0 18px rgba(224,103,63,.45))}
.pill{display:inline-block;padding:.25rem .85rem;border-radius:99px;border:1px solid currentColor;font-size:.78rem;letter-spacing:.14em;font-weight:700}
.note{background:rgba(217,180,90,.10);border-left:3px solid #D9B45A;padding:.8rem 1rem;border-radius:6px;font-size:.92rem;color:#d8cdbd}
div[data-testid="stMetric"]{background:rgba(255,255,255,.05);border:1px solid rgba(255,255,255,.1);border-radius:16px;padding:.8rem 1rem}
.stButton>button,.stDownloadButton>button{border-radius:99px;background:linear-gradient(90deg,#D9B45A,#E0673F);color:#0E0B09;border:0;padding:.5rem 1.5rem;font-weight:700}
button[data-baseweb="tab"]{font-weight:600}
</style>""", unsafe_allow_html=True)

def hero(kicker, title, sub, decor=True):
    d = (svg("plate", 190, "right:6%;top:12%") + svg("coin", 90, "right:30%;top:58%;animation-delay:-2s") +
         svg("leaf", 80, "right:2%;bottom:6%;animation-delay:-4s") + svg("cup", 110, "right:24%;top:6%;animation-delay:-1s")) if decor else ""
    st.markdown(f'<div class="hero2">{d}<div class="k">{kicker}</div><h1>{title}</h1><p>{sub}</p></div>', unsafe_allow_html=True)

def card(col, n, l): col.markdown(f'<div class="gc"><b class="n">{n}</b><span class="l">{l}</span></div>', unsafe_allow_html=True)

def style(fig, h=340):
    fig.update_layout(height=h, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,.03)", margin=dict(l=10, r=10, t=40, b=10),
                      font=dict(family="DM Sans", color=INK), colorway=[CLAY, GOLD, MOSS, "#7A6F60"], title_font=dict(family="Fraunces", size=17))
    fig.update_xaxes(gridcolor='rgba(255,255,255,.08)', zeroline=False); fig.update_yaxes(gridcolor='rgba(255,255,255,.08)', zeroline=False); return fig

@st.cache_resource(show_spinner="Training and cross-validating models…")
def load():
    raw = pl.load_raw(); A = pl.get_artifacts(); X, y = pl.preprocess(raw); return raw, X, y, A


def site_inputs(key, X, imp, A):
    grp = st.radio("City group", ["Big Cities", "Other"], horizontal=True, key=key + "g")
    typ = st.selectbox("Restaurant type", ["FC (Food Court)", "IL (Inline)", "DT (Drive-Thru)", "MB (Mobile)"], key=key + "t")
    days = st.slider("Days since opening", 0, 8000, int(X.days_open.median()), 30, key=key + "d")
    row = A["medians"].copy(); row["days_open"], row["is_big_city"] = days, int(grp == "Big Cities")
    for t in ["FC", "IL", "DT", "MB"]: row[f"type_{t}"] = int(typ.startswith(t))
    for k in [i for i in imp.index if i.startswith("P")][:4]:
        row[k] = st.slider(lab(k), 0.0, float(X[k].max()), float(X[k].median()), key=key + k)
    st.caption("Indicator names are illustrative labels for anonymised data.")
    return row

def predict(A, rows):
    return A["model"].predict(pd.DataFrame(rows)[A["columns"]])


@st.cache_resource(show_spinner="Training attendance models (first run takes a few minutes)…")
def load_att():
    import attendance as at
    return at.get_artifacts()
