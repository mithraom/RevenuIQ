"""RevenuIQ attendance module: daily visitor forecasting per restaurant.
Data: Kaggle 'Recruit Restaurant Visitor Forecasting' (air_visit_data, air_store_info, date_info, air_reserve) in data/attendance/.
Design: every history feature is lagged by >= 7 days, so the model is a genuine up-to-7-day-ahead forecast (no look-ahead).
Validation is time-ordered (train on the past, test on the following 28 days), never random."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, joblib
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

D = Path(__file__).parent / "data" / "attendance"
MODEL = Path(__file__).parent / "attendance_model.joblib"
H = 7
FEATS = ["dow", "month", "dom", "weekend", "holiday_flg", "prev_h", "next_h", "genre", "area", "lat", "lon", "reserved_7d",
         "lag7", "lag14", "lag21", "lag28", "same_dow_mean", "roll28", "roll_med28", "prior_mean"]
B1, B2 = "Baseline: same weekday last week", "Baseline: 28-day store mean"


def available(): return all((D / f).exists() for f in ["air_visit_data.csv", "air_store_info.csv", "date_info.csv"])


def load():
    v = pd.read_csv(D / "air_visit_data.csv", parse_dates=["visit_date"]); s = pd.read_csv(D / "air_store_info.csv")
    h = pd.read_csv(D / "date_info.csv", parse_dates=["calendar_date"]).rename(columns={"calendar_date": "visit_date"})
    r = None
    if (D / "air_reserve.csv").exists():  # only reservations booked >= 7 days ahead are known at forecast time
        rr = pd.read_csv(D / "air_reserve.csv", parse_dates=["visit_datetime", "reserve_datetime"])
        rr = rr[(rr.visit_datetime - rr.reserve_datetime) >= pd.Timedelta(days=H)]; rr["visit_date"] = rr.visit_datetime.dt.normalize()
        r = rr.groupby(["air_store_id", "visit_date"]).reserve_visitors.sum().rename("reserved_7d").reset_index()
    return v, s, h, r


def build_panel(v, s, h, r):
    last = v.visit_date.max(); dates = pd.date_range(v.visit_date.min(), last + pd.Timedelta(days=H))
    wide = v.pivot_table(index="visit_date", columns="air_store_id", values="visitors", aggfunc="sum").reindex(dates)
    base = wide.shift(H)
    f = {f"lag{k}": wide.shift(k) for k in (7, 14, 21, 28)}
    f["roll28"] = base.rolling(28, min_periods=7).mean(); f["roll_med28"] = base.rolling(28, min_periods=7).median()
    f["prior_mean"] = base.expanding(min_periods=7).mean(); f["visitors"] = wide
    idx = pd.MultiIndex.from_product([dates, wide.columns], names=["visit_date", "air_store_id"])
    p = pd.DataFrame({k: x.to_numpy().ravel() for k, x in f.items()}, index=idx).reset_index()
    p["same_dow_mean"] = p[["lag7", "lag14", "lag21", "lag28"]].mean(axis=1)
    d = p.visit_date.dt; p["dow"], p["month"], p["dom"] = d.dayofweek, d.month, d.day; p["weekend"] = (p.dow >= 5).astype(int)
    h = h.sort_values("visit_date").assign(prev_h=lambda x: x.holiday_flg.shift(1).fillna(0), next_h=lambda x: x.holiday_flg.shift(-1).fillna(0))
    p = p.merge(h[["visit_date", "holiday_flg", "prev_h", "next_h"]], on="visit_date", how="left")
    s = s.assign(prefecture=s.air_area_name.str.split().str[0]); s["genre"] = s.air_genre_name.astype("category").cat.codes
    s["area"] = s.prefecture.astype("category").cat.codes; s = s.rename(columns={"latitude": "lat", "longitude": "lon"})
    p = p.merge(s[["air_store_id", "genre", "area", "lat", "lon"]], on="air_store_id", how="left")
    p = p.merge(r, on=["air_store_id", "visit_date"], how="left") if r is not None else p.assign(reserved_7d=0)
    p["reserved_7d"] = p.reserved_7d.fillna(0); return p, s, last


def make(name, params=None):
    if name == "Ridge": e = Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler()), ("m", RidgeCV(alphas=np.logspace(-1, 3, 15)))])
    elif name == "Random Forest": e = Pipeline([("i", SimpleImputer(strategy="median")), ("m", RandomForestRegressor(100, min_samples_leaf=10, max_features=.5, max_samples=.4, n_jobs=-1, random_state=42))])
    elif name == "Hist Gradient Boosting": e = HistGradientBoostingRegressor(max_iter=200, learning_rate=.08, max_depth=6, min_samples_leaf=40, random_state=42)
    else:
        from xgboost import XGBRegressor
        e = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=.05, subsample=.8, colsample_bytree=.8, random_state=42, n_jobs=-1)
    return e.set_params(**params) if params else e


def model_names():
    n = ["Ridge", "Random Forest", "Hist Gradient Boosting"]
    try: import xgboost; n.append("XGBoost")
    except Exception: pass
    return n


GRIDS = {"Hist Gradient Boosting": {"learning_rate": [.05, .1], "max_depth": [4, 8]}, "XGBoost": {"learning_rate": [.03, .08], "max_depth": [4, 8]}}


def search(name): return GridSearchCV(make(name), GRIDS[name], cv=TimeSeriesSplit(3), scoring="neg_root_mean_squared_error", n_jobs=1)


def metrics(y, p):
    p = np.clip(p, 0, None)
    return dict(RMSLE=float(np.sqrt(np.mean((np.log1p(p) - np.log1p(y)) ** 2))), RMSE=float(np.sqrt(np.mean((p - y) ** 2))),
                MAE=float(np.mean(np.abs(p - y))), R2=float(r2_score(y, p)))


def folds(d, last):
    out = []
    for c in (last - pd.Timedelta(days=55), last - pd.Timedelta(days=27)):
        out.append((d[d.visit_date < c], d[(d.visit_date >= c) & (d.visit_date < c + pd.Timedelta(days=28))]))
    return out


def fitpred(est, tr, te):
    est.fit(tr[FEATS], np.log1p(tr.visitors)); return np.expm1(np.clip(est.predict(te[FEATS]), 0, np.log1p(tr.visitors.max())))  # clip stops extrapolation blow-ups


def baseline(te, tr, kind):
    fb = tr.visitors.median(); a = te.roll28.fillna(te.prior_mean).fillna(fb)
    return te.lag7.fillna(a) if kind == B1 else a


def avg(rows, name): return {"Model": name, **pd.DataFrame(rows).mean().to_dict()}


def rank(rows):
    d = pd.DataFrame(rows); b = d[d.Model.str.startswith("Baseline")]
    return pd.concat([d[~d.Model.str.startswith("Baseline")].sort_values("RMSLE"), b.sort_values("RMSLE")]).reset_index(drop=True)


def train_and_save():
    v, s, h, r = load(); p, s, last = build_panel(v, s, h, r); d = p[p.visitors.notna()].reset_index(drop=True); fs = folds(d, last)
    rows = [avg([metrics(te.visitors, baseline(te, tr, k)) for tr, te in fs], k) for k in (B1, B2)]
    for n in model_names():
        rows.append(avg([metrics(te.visitors, fitpred(make(n), tr, te)) for tr, te in fs], n)); print("done", n, flush=True)
    res = rank(rows); best = res.loc[0, "Model"]; tuning, params = None, None
    if best in GRIDS:
        print("tuning", best, flush=True)
        row = avg([metrics(te.visitors, fitpred(search(best), tr, te)) for tr, te in fs], f"{best} (tuned)")
        g = search(best).fit(d[FEATS], np.log1p(d.visitors)); tuning = dict(model=best, params=g.best_params_, before=res.loc[0].to_dict(), after=row)
        res = rank(res.to_dict("records") + [row])
        if row["RMSLE"] < res.loc[res.Model == best, "RMSLE"].iloc[0]: params = g.best_params_
    top = res.loc[0, "Model"]; name = top.replace(" (tuned)", ""); final = make(name, params).fit(d[FEATS], np.log1p(d.visitors))
    tr, te = fs[-1]; bm = make(name, params).fit(tr[FEATS], np.log1p(tr.visitors)); pr = np.expm1(np.clip(bm.predict(te[FEATS]), 0, np.log1p(tr.visitors.max())))
    bt = te[["air_store_id", "visit_date", "visitors", "dow"]].assign(pred=np.clip(pr, 0, None)).reset_index(drop=True)
    smp = te.sample(min(8000, len(te)), random_state=0)
    pi = permutation_importance(bm, smp[FEATS], np.log1p(smp.visitors), n_repeats=3, random_state=0, scoring="neg_root_mean_squared_error")
    imp = pd.Series(pi.importances_mean, FEATS).sort_values(ascending=False)
    fc = p[p.visit_date > last].reset_index(drop=True); fc["forecast"] = np.expm1(np.clip(final.predict(fc[FEATS]), 0, np.log1p(d.visitors.max())))
    hist = d[["air_store_id", "visit_date", "visitors"]]
    joblib.dump({"results": res, "best": top, "tuning": tuning, "importance": imp, "backtest": bt, "forecast": fc[["air_store_id", "visit_date", "dow", "holiday_flg", "reserved_7d", "forecast"]],
                 "history": hist, "stores": s[["air_store_id", "air_genre_name", "air_area_name", "prefecture", "lat", "lon"]], "last": last,
                 "rmsle": float(res.loc[0, "RMSLE"]), "holidays": h}, MODEL)
    return joblib.load(MODEL)


def get_artifacts(): return joblib.load(MODEL) if MODEL.exists() else train_and_save()


if __name__ == "__main__":
    a = train_and_save(); print(a["results"].round(3).to_string(index=False)); print("Best:", a["best"])
    if a["tuning"]: print("Tuned", a["tuning"]["model"], "->", a["tuning"]["params"])
