# RevenuIQ: Restaurant Revenue Prediction and Site Evaluation

Course project for **22MDC56 Machine Learning Laboratory**
Author: Mithra Omprakash

RevenuIQ predicts the annual revenue of a restaurant branch and turns the forecast into a site decision (**OPEN / REVIEW / AVOID**). It is built on the Kaggle *TFI Restaurant Revenue Prediction* dataset and delivered as an interactive Streamlit app.

---

## 1. Problem statement

Opening a restaurant in the wrong location is costly. Chains need a defensible estimate of what a candidate site will earn, and a way to compare candidate sites. This project asks:

> Given a site's location type, restaurant format, age and a set of anonymised demographic and real-estate indicators, how much revenue will it earn, and is the site worth opening?

## 2. Research gaps addressed

The literature survey (10 papers on restaurant revenue and attendance prediction) identified three gaps. Each maps to something in this project:

| Gap in existing work | How RevenuIQ addresses it |
|---|---|
| Studies apply regression, tree ensembles or deep learning individually and rarely combine dimensionality reduction with a full model comparison | Correlation filtering and PCA are built into the pipeline, and eight models are compared under the same cross-validation |
| Many papers stop at model accuracy and give no usable tool | A multi-page Streamlit app turns each prediction into a verdict with supporting evidence |
| The TFI dataset has 37 correlated predictors but only 137 training rows, which invites overfitting | PCA and correlation filtering are fitted **inside each cross-validation fold**, so there is no data leakage |

## 3. Dataset

- **Source:** Kaggle competition *Restaurant Revenue Prediction* (TFI).
- **File used:** `train.csv` only. It has 137 branches with `Id`, `Open Date`, `City`, `City Group`, `Type`, 37 anonymised indicators `P1` to `P37`, and `revenue`.
- `test.csv` (no revenue column) and `sampleSubmission.csv` are not used.
- **The data is not included in this repository.** Kaggle's competition rules restrict re-sharing. Download it yourself (section 6).

## 3b. Attendance data (module 2)
Kaggle competition *Recruit Restaurant Visitor Forecasting*: download `air_visit_data.csv`, `air_store_info.csv`, `date_info.csv` and `air_reserve.csv` and put them in `data/attendance/` (not committed). 829 restaurants, about 252,000 daily visit records from Jan 2016 to Apr 2017.

**Attendance method (`attendance.py`):** every history feature (visits 7/14/21/28 days ago, rolling 28-day mean and median, store prior mean) is lagged by at least 7 days, so the model is a genuine up-to-7-day-ahead forecast. Holidays, cuisine, region, location and reservations booked at least 7 days ahead are added. Models (Ridge, Random Forest, Hist Gradient Boosting, XGBoost) are compared against two baselines with **time-ordered validation** (two future 4-week windows, never shuffled), scored by RMSLE, RMSE, MAE and R². The best model is tuned with a time-series grid search. Run `py attendance.py` to train and print the table.

## 4. How it works

### 4.1 Preprocessing and feature engineering (`pipeline.preprocess`)
- `Open Date` becomes **days open**, counted from 23 March 2015.
- `City Group` becomes a 0/1 column. `Type` is one-hot encoded.
- `City` is dropped because it has too many distinct values for 137 rows.
- Revenue is right-skewed, so models are trained on **log revenue** and predictions are converted back to dollars.
- Many P-variables contain a lot of zeros that are probably missing values. They are kept as given and handled by scaling and PCA.

### 4.2 Leakage-free feature reduction
Inside every cross-validation fold:
1. `StandardScaler` standardises features.
2. `CorrFilter` drops one feature from every pair with |correlation| above 0.9.
3. `PCA` keeps components that explain 95% of the variance.

### 4.3 Models compared (`pipeline.models`)
| Model | Purpose |
|---|---|
| Baseline (mean) | Reference: always predicts average revenue |
| Linear Regression | Simple linear benchmark |
| Linear Regression + PCA | Tests dimensionality reduction on a linear model |
| Random Forest | Tree ensemble on all features |
| Random Forest (tuned) | Nested grid search over depth and leaf size |
| PCA + Random Forest | The pipeline proposed in the literature |
| SVR + PCA | Kernel method |
| XGBoost | Boosted trees (falls back to Gradient Boosting if XGBoost is not installed) |

### 4.4 Evaluation
5-fold cross-validation repeated 3 times. Metrics are **RMSE**, **MAE** and **R²**. The best real model (the baseline is ranked last and never selected) is refitted on all data and saved as `model.joblib`.

## 5. The app

| Page | What it does |
|---|---|
| **Home** | Problem statement, research gaps and headline numbers |
| **Site Evaluator** | Describe a site and get a revenue forecast with range, percentile, OPEN/REVIEW/AVOID verdict, 5 comparable real branches, lever sensitivity, a maturity curve and a downloadable site report |
| **Compare & Shortlist** | Site A versus Site B, or upload a CSV of candidate sites to rank them and download predictions |
| **Attendance Forecast** | Pick a restaurant: 7-day visitor forecast with range, backtest vs actual, network insights (weekday, holiday, cuisine, region) and a model lab |
| **Analytics** | Interactive EDA: distributions, cities and types, maturity, correlations, 3D PCA map, feature importance |
| **Model Lab** | Model comparison table with error bars and out-of-fold predicted vs actual |

**How the verdict works:** the forecast is ranked against existing branches. At or above the 66th percentile is OPEN, 33rd to 66th is REVIEW, and below is AVOID. These cutoffs are illustrative and should be calibrated with real business rules.

## 6. Setup and run

### Requirements
Python 3.10 or newer.

```bash
py -m pip install -r requirements.txt
```

### Get the data
Option A: download manually from the Kaggle competition's Data tab, put `train.csv` in the `data/` folder.
Option B: `py download_data.py` (needs a Kaggle API token and having joined the competition).
Option C: upload `train.csv` in the app's sidebar.

If the hidden file `data/.synthetic` exists, the app is running on **fake demo data**. Delete it once the real file is in place.

### Run the app
```bash
py -m streamlit run app.py
```
Opens at `http://localhost:8501`. The first launch trains and cross-validates all models (about a minute) and caches `model.joblib`. Use **Retrain models** in the sidebar after changing the data.

### Generate the report, slides and viva notes
```bash
py make_deliverables.py
```
Creates `deliverables/` with `RevenuIQ_Report.docx`, `RevenuIQ_Slides.pptx`, `viva_notes.md` and `figures/`. All numbers come from your real results.

## 7. Project structure

```
revenuiq/
├── app.py                  # Streamlit entry point and navigation
├── pipeline.py             # preprocessing, models, cross-validation, data summary
├── attendance.py           # attendance data prep, models, time-ordered validation
├── ui.py                   # shared styling and helper functions
├── views/
│   ├── home.py
│   ├── evaluate.py         # Site Evaluator
│   ├── footfall.py         # Attendance Forecast page
│   ├── compare.py          # Compare & Shortlist
│   ├── analytics.py
│   └── models.py           # Model Lab
├── make_deliverables.py    # builds report, slides, viva notes
├── download_data.py        # fetches train.csv from Kaggle
├── make_demo_data.py       # synthetic stand-in data (demo only)
├── requirements.txt
└── data/                   # train.csv goes here (not committed)
```

## 8. Results

Fill this in from your own run (`py pipeline.py` prints the table, and the Model Lab page shows it).

| Model | RMSE ($M) | MAE ($M) | R² |
|---|---|---|---|
| Best model: _name_ | _ | _ | _ |
| Baseline (mean) | _ | _ | _ |

Report the numbers as measured. On a dataset this small and noisy, a modest R² is expected.

## 9. Limitations

- Only 137 training rows against 43 features: cross-validation scores vary a lot between folds.
- P1 to P37 are anonymised, so their business meaning cannot be explained.
- Zero values in many indicators are kept as given, not imputed.
- Outlier branches are kept because they are real observations.
- The OPEN/REVIEW/AVOID cutoffs are heuristic, and there is no external validation set.
- Forecasts are best used to **rank** sites, not to set financial commitments.

## 10. Future work

- Link the two modules: the revenue and attendance datasets come from different chains and countries, so they are separate models.
- Add weather and social-media signals suggested by the literature.
- Calibrate prediction intervals and the verdict thresholds.
- Test Ridge/Elastic Net, missing-value flags and model ensembling to improve R².
- Deploy online with a feedback loop for retraining.

## 11. Tech stack

Python, pandas, NumPy, scikit-learn, XGBoost, Plotly, Streamlit, Matplotlib, python-docx, python-pptx.

## 12. References

1. C. Lan et al., "Research on the Prediction System of Event Attendance in an Event-Based Social Network," *Wireless Communications and Mobile Computing*, 2022.
2. Nguyen et al., "Predicting Individual Event Attendance with Machine Learning: A Step-Forward Approach," *Applied Economics* 54(27), 2022.
3. J.L. Ooi and R. Mafas, "A Review on the Predictive Algorithms for Event Attendance Prediction in Event-Based Social Networks."
4. "Event Attendance Prediction Using Social Media," ACM Digital Library, 2020.
5. US Patents 11,790,386 / 12,288,220, "Machine-Learned Attendance Prediction for Ticket Distribution."
6. "Predicting Restaurant Revenue using Machine Learning," JCIT.
7. "Restaurant Revenue Prediction Using Machine Learning: A Comparative Study of Multiple Linear Regression and Random Forest Models," ECAI, 2024.
8. S. Gogolev and E. Sotnikov, "Comparison of Machine Learning Algorithms in Restaurant Revenue Prediction," 2020.
9. G.P.S. Rao et al., "Machine Learning based Restaurant Revenue Prediction," Springer AISC, 2021.
10. G.S. Rao, K.A. Shastry, S. Sathyashree et al., "Restaurant Revenue Prediction Applying Supervised Learning Methods."

Dataset: Kaggle, *TFI Restaurant Revenue Prediction* competition.
