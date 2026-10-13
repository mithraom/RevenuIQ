"""Synthetic stand-in with the same schema as Kaggle TFI train.csv (137 rows, P1-P37).
Use ONLY if you cannot download the real file. Real data: kaggle.com/c/restaurant-revenue-prediction"""
import numpy as np, pandas as pd
rng = np.random.default_rng(7)
n = 137
cities = ["İstanbul", "Ankara", "İzmir", "Bursa", "Samsun", "Antalya", "Sakarya", "Adana", "Trabzon", "Kayseri"]
city = rng.choice(cities, n, p=[.3, .1, .1, .08, .08, .08, .07, .07, .06, .06])
big = np.isin(city, ["İstanbul", "Ankara", "İzmir"])
df = pd.DataFrame({"Id": range(n)})
df["Open Date"] = pd.to_datetime("1996-01-01") + pd.to_timedelta(rng.integers(0, 365 * 19, n), "D")
df["City"], df["City Group"] = city, np.where(big, "Big Cities", "Other")
df["Type"] = rng.choice(["FC", "IL", "DT"], n, p=[.55, .43, .02])
base = rng.integers(1, 8, (n, 37)).astype(float)
for c in list(range(13, 18)) + list(range(23, 27)) + list(range(29, 37)):
    base[rng.random(n) < .45, c] = 0
for i in range(37): df[f"P{i+1}"] = base[:, i]
age = (pd.Timestamp("2015-03-23") - df["Open Date"]).dt.days / 365
z = 0.35*np.log1p(age) + 0.3*df["P29"].clip(1)/3 + 0.12*df["P6"] + 0.15*big + 0.1*df["P2"] + rng.normal(0, .25, n)
df["revenue"] = (3.2e6 * np.exp(z - z.mean())).round()
df.loc[rng.choice(n, 3, replace=False), "revenue"] *= 3.2
df.to_csv("data/train.csv", index=False); open("data/.synthetic", "w").close(); print("wrote data/train.csv (SYNTHETIC)")
