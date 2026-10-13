"""Fetch the real Kaggle TFI train.csv. One-time setup:
1. Accept the rules on kaggle.com/c/restaurant-revenue-prediction (Join Competition)
2. kaggle.com > Settings > Create New Token -> save kaggle.json to ~/.kaggle/
3. pip install kaggle && python download_data.py"""
import subprocess, zipfile, pathlib
d = pathlib.Path("data"); d.mkdir(exist_ok=True)
subprocess.run(["kaggle", "competitions", "download", "-c", "restaurant-revenue-prediction", "-p", str(d)], check=True)
z = next(d.glob("*.zip"))
with zipfile.ZipFile(z) as f: f.extract("train.csv.zip" if "train.csv.zip" in f.namelist() else "train.csv", d)
for p in d.glob("train.csv.zip"): zipfile.ZipFile(p).extractall(d)
(d / ".synthetic").unlink(missing_ok=True); print("Real train.csv ready:", d / "train.csv")
