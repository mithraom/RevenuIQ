"""RevenuIQ backend: preprocessing, leakage-free model comparison, tuning of the best model."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, joblib
from pathlib import Path
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin, clone
from sklearn.decomposition import PCA
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.linear_model import ElasticNetCV, LinearRegression, RidgeCV
from sklearn.metrics import make_scorer, r2_score
from sklearn.model_selection import GridSearchCV, RepeatedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

DATA = Path(__file__).parent / "data" / "train.csv"
MODEL = Path(__file__).parent / "model.joblib"
REF_DATE = pd.Timestamp("2015-03-23")  # reference date used in the TFI competition
PCOLS = [f"P{i}" for i in range(1, 38)]
BASE = "Baseline (mean)"


# ------------------------------------------------------------------ data
def load_raw():
    return pd.read_csv(DATA, parse_dates=["Open Date"], date_format="%m/%d/%Y") if DATA.exists() else None


def preprocess(df):
    """Clean + engineer features. Returns X (numeric DataFrame) and y (or None)."""
    d = df.copy()
    od = pd.to_datetime(d["Open Date"], errors="coerce", format="mixed")
    d["days_open"] = (REF_DATE - od).dt.days
    d["is_big_city"] = (d["City Group"] == "Big Cities").astype(int)
    for t in ["FC", "IL", "DT", "MB"]:
        d[f"type_{t}"] = (d["Type"] == t).astype(int)
    X = d[PCOLS + ["days_open", "is_big_city", "type_FC", "type_IL", "type_DT", "type_MB"]].astype(float)
    return X, (d["revenue"] if "revenue" in d else None)


# ------------------------------------------------------------------ custom steps (all fitted inside CV folds)
class ZeroFix(BaseEstimator, TransformerMixin):
    """P-columns with many zeros are treated as missing: add a 0/1 'was missing' flag and fill with the median of non-zero values."""
    def __init__(self, thr=0.3): self.thr = thr
    def fit(self, X, y=None):
        X = pd.DataFrame(X); self.cols_ = list(X.columns)
        self.zc_ = [c for c in PCOLS if c in X.columns and (X[c] == 0).mean() > self.thr]
        self.med_ = {c: (X.loc[X[c] != 0, c].median() if (X[c] != 0).any() else 0.0) for c in self.zc_}
        return self
    def transform(self, X):
        X = pd.DataFrame(X, columns=self.cols_).copy()
        for c in self.zc_:
            z = X[c] == 0; X[c + "_miss"] = z.astype(float); X.loc[z, c] = self.med_[c]
        return X


class CorrFilter(BaseEstimator, TransformerMixin):
    """Drops one of every feature pair with |r| above threshold."""
    def __init__(self, threshold=0.9): self.threshold = threshold
    def fit(self, X, y=None):
        with np.errstate(all="ignore"): c = np.abs(np.corrcoef(np.asarray(X), rowvar=False))
        c = np.nan_to_num(c); keep = []
        for i in range(c.shape[0]):
            if all(c[i, j] <= self.threshold for j in keep): keep.append(i)
        self.keep_ = keep; return self
    def transform(self, X): return np.asarray(X)[:, self.keep_]


class LogCap(BaseEstimator, RegressorMixin):
    """Trains on log(revenue) with the target capped at a training-fold quantile (reduces outlier pull),
    clips predictions to the training range (stops extrapolation blow-ups), and returns dollars.
    Evaluation is always against the original, uncapped revenue."""
    def __init__(self, est=None, cap_q=0.97): self.est, self.cap_q = est, cap_q
    def fit(self, X, y):
        y = np.asarray(y, float); self.cap_ = np.quantile(y, self.cap_q); z = np.log1p(np.minimum(y, self.cap_))
        self.lo_, self.hi_ = z.min() - .25, z.max() + .25
        self.est_ = clone(self.est).fit(X, z); return self
    def predict(self, X): return np.expm1(np.clip(self.est_.predict(X), self.lo_, self.hi_))


# ------------------------------------------------------------------ models
def _xgb():
    try:
        from xgboost import XGBRegressor
        return "XGBoost", XGBRegressor(n_estimators=200, max_depth=2, learning_rate=.03, subsample=.7, colsample_bytree=.6,
                                       min_child_weight=5, reg_lambda=5, random_state=42, n_jobs=1)
    except Exception:
        from sklearn.ensemble import GradientBoostingRegressor
        return "Gradient Boosting", GradientBoostingRegressor(n_estimators=150, max_depth=2, learning_rate=.03, subsample=.7, random_state=42)


def _rf(): return RandomForestRegressor(200, max_depth=4, min_samples_leaf=5, max_features=.5, random_state=42, n_jobs=-1)
def _pre(): return [("fix", ZeroFix()), ("scale", StandardScaler())]
def _red(): return _pre() + [("corr", CorrFilter(0.9)), ("pca", PCA(0.95))]


def build():
    """Raw (log-scale) estimators. Every pipeline starts with ZeroFix so zero-as-missing handling is fitted per fold."""
    gbn, gb = _xgb()
    return {
        "Linear Regression": Pipeline(_pre() + [("m", LinearRegression())]),
        "Ridge": Pipeline(_pre() + [("m", RidgeCV(alphas=np.logspace(-1, 3, 20)))]),
        "Elastic Net": Pipeline(_pre() + [("m", ElasticNetCV(l1_ratio=[.2, .5, .8], cv=5, max_iter=20000))]),
        "Linear Regression + PCA": Pipeline(_red() + [("m", LinearRegression())]),
        "Random Forest": Pipeline([("fix", ZeroFix()), ("m", _rf())]),
        "PCA + Random Forest": Pipeline(_red() + [("m", _rf())]),
        "SVR + PCA": Pipeline(_red() + [("m", SVR(C=3, epsilon=0.1))]),
        gbn: Pipeline([("fix", ZeroFix()), ("m", gb)]),
        "Ensemble (Ridge+RF+Boost)": VotingRegressor([
            ("ridge", Pipeline(_pre() + [("m", RidgeCV(alphas=np.logspace(-1, 3, 20)))])),
            ("rf", Pipeline([("fix", ZeroFix()), ("m", _rf())])), ("gb", Pipeline([("fix", ZeroFix()), ("m", _xgb()[1])]))]),
    }


GRIDS = {  # small grids (8 combos or fewer) so nested tuning stays fast; used on whichever model ranks best
    "Random Forest": {"m__max_depth": [3, 5], "m__min_samples_leaf": [3, 8], "m__max_features": [.3, .6]},
    "PCA + Random Forest": {"pca__n_components": [.9, .95], "m__max_depth": [3, 5], "m__min_samples_leaf": [3, 8]},
    "SVR + PCA": {"m__C": [.3, 1, 3], "m__epsilon": [.1, .2]},
    "XGBoost": {"m__max_depth": [2, 3], "m__learning_rate": [.03, .06], "m__n_estimators": [150, 300]},
    "Gradient Boosting": {"m__max_depth": [2, 3], "m__learning_rate": [.03, .06]},
}


def models():
    m = {BASE: DummyRegressor()}
    m.update({n: LogCap(e) for n, e in build().items()}); return m


def tuned(name):
    return LogCap(GridSearchCV(build()[name], GRIDS[name], cv=3, scoring="neg_root_mean_squared_error", n_jobs=1))


def get_model(name):
    return tuned(name[:-8]) if name.endswith(" (tuned)") else models()[name]


# ------------------------------------------------------------------ evaluation
def _r2log(yt, yp): return r2_score(np.log1p(yt), np.log1p(np.clip(yp, 0, None)))


def _score(name, est, X, y):
    cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)
    sc = {"rmse": "neg_root_mean_squared_error", "mae": "neg_mean_absolute_error", "r2": "r2", "r2log": make_scorer(_r2log)}
    r = cross_validate(est, X, y, cv=cv, scoring=sc)
    return {"Model": name, "RMSE": -r["test_rmse"].mean(), "RMSE_sd": r["test_rmse"].std(), "MAE": -r["test_mae"].mean(),
            "R2": r["test_r2"].mean(), "R2_log": r["test_r2log"].mean()}


def _rank(rows):
    d = pd.DataFrame(rows); b = d[d.Model == BASE]
    return pd.concat([d[d.Model != BASE].sort_values("RMSE"), b]).reset_index(drop=True)  # real models ranked, baseline last


def compare(X, y):
    return _rank([_score(n, m, X, y) for n, m in models().items()])


def summary(raw, X, y):
    from scipy.stats import skew
    sc = StandardScaler().fit_transform(X); q1, q3 = y.quantile([.25, .75]); ev = np.cumsum(PCA().fit(sc).explained_variance_ratio_)
    return dict(rows=len(X), features=X.shape[1], missing=int(raw.isna().sum().sum()), zero_heavy=int(((X[PCOLS] == 0).mean() > .3).sum()),
                outliers=int((y > q3 + 1.5 * (q3 - q1)).sum()), skew_raw=float(skew(y)), skew_log=float(skew(np.log1p(y))),
                after_corr=len(CorrFilter(0.9).fit(sc).keep_), pca95=int((ev < .95).sum() + 1),
                mean=float(y.mean()), median=float(y.median()), min=float(y.min()), max=float(y.max()))


def train_and_save():
    X, y = preprocess(load_raw())
    res = compare(X, y); best = res.loc[0, "Model"]; tuning = None
    if best in GRIDS:  # tune ONLY the best untuned model, nested inside the outer CV so the score is honest
        t = tuned(best); row = _score(f"{best} (tuned)", t, X, y); before = res.loc[0].to_dict()
        t.fit(X, y); tuning = dict(model=best, params=t.est_.best_params_, before=before, after=row)
        res = _rank(res.to_dict("records") + [row])
    top = res.loc[0, "Model"]; final = get_model(top).fit(X, y)
    cap = y.quantile(.97)
    imp = pd.Series(RandomForestRegressor(300, max_depth=5, min_samples_leaf=3, random_state=42, n_jobs=-1)
                    .fit(X, np.log1p(np.minimum(y, cap))).feature_importances_, X.columns).sort_values(ascending=False)
    ev = PCA().fit(StandardScaler().fit_transform(X)).explained_variance_ratio_
    joblib.dump({"model": final, "best": top, "results": res, "importance": imp, "pca_var": ev, "tuning": tuning,
                 "medians": X.median(), "columns": list(X.columns)}, MODEL)
    return joblib.load(MODEL)


def get_artifacts():
    return joblib.load(MODEL) if MODEL.exists() else train_and_save()


if __name__ == "__main__":
    import pipeline as _p  # re-import so custom classes are saved as pipeline.LogCap (not __main__.LogCap), which the app can load
    a = _p.train_and_save(); print(a["results"].round(3).to_string(index=False)); print("Best:", a["best"])
    if a["tuning"]: print("Tuned", a["tuning"]["model"], "->", a["tuning"]["params"])