# ExoBias – Exoplanet Uncertainty Analysis System

**An Uncertainty-Aware Machine Learning Framework for Investigating the Impact of Candidate-Label
Uncertainty on Exoplanet Population Analysis**

ExoBias asks: *how do different strategies for handling uncertain Kepler **Candidate** labels affect
machine-learning reliability and the interpretation of observed exoplanet population patterns?*
It trains real models on the NASA Kepler DR25 KOI table you upload, compares an ANN against traditional
models, runs population and robustness analyses, and reports what the data actually show — it never
assumes the ANN wins or that confidence-weighting helps.

---------------------------------------------------------------------------------------------------

## 1. Setup (VS Code terminal)

```bash
cd ExoBias
python -m venv .venv
# Windows (PowerShell):   .venv\Scripts\Activate.ps1
# Windows (cmd):          .venv\Scripts\activate.bat
# macOS / Linux:          source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip freeze > requirements.lock.txt     # record the exact environment that worked
```

Use **Python 3.11 or 3.12** if you can (TensorFlow wheels). XGBoost is optional; skip it with
`pip install -r requirements.txt` edited, or ignore if it fails — the app detects it automatically.

## 2. Run

```bash
streamlit run ExoBias.py          # the dashboard
pytest -q                     # automated tests
```

Pages (sidebar order): Project Overview → Data Explorer → Data Preprocessing → ANN Workbench →
Model Comparison → Population Analysis → Uncertainty Analysis → Results Report.
**Nothing trains until you click a button**, and results persist while you navigate (Streamlit session state).

## 3. Dataset

**Upload the Kepler DR25 KOI table from the NASA Exoplanet Archive** (table name `q1_q17_dr25_koi`).

* Direct CSV download (paste into your browser):
  `https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+*+from+q1_q17_dr25_koi&format=csv`
* If the saved file has no `.csv` extension, rename it (for example `koi_dr25.csv`).
* Manual route: <https://exoplanetarchive.ipac.caltech.edu/> → Data → Kepler Objects of Interest → **Q1–Q17 DR25** → Download Table → CSV (all columns).
* Upload it on the **Data Explorer** page (the page also shows these download links), or drop it into `data/`.

The table name and the TAP `&format=csv` URL pattern come from NASA's own documentation. The file itself could not be downloaded
while developing this project, so the app validates whatever you upload and tells you exactly which columns or classes are missing.

| Field | Use |
|---|---|
| `koi_period, koi_prad, koi_teq, koi_insol` | planetary model inputs |
| `koi_steff, koi_srad, koi_smass` | host-star model inputs |
| `koi_disposition` | target (CANDIDATE / FALSE POSITIVE; CONFIRMED only if genuinely present) |
| `koi_score` | **only** for Experiments B and C – never a model input |
| `kepid` | host-star id, used **only** for grouped splitting |

DR25 mainly has CANDIDATE and FALSE POSITIVE. ExoBias displays the classes present before training and enables the
three-class experiment only if all three exist. Only columns actually in your file are used. `koi_score`,
`kepid`, names, dispositions and `koi_fpflag_*` are blocked as inputs (enforced in code and tested).

**`koi_score` is a Robovetter vetting score, not a calibrated probability that an object is a real planet.**

## 4. What the pipeline does

| Stage | Details |
|---|---|
| Cleaning | removes unsupported labels, duplicates (by `kepoi_name`), all-missing rows; infinities → NaN; **row counts reported per stage** |
| Split | ~70/15/15, **grouped by `kepid`** (verified: 0 shared stars); stratified row split available for comparison; warned fallback if impossible |
| Leakage control | median imputer + `StandardScaler` fitted on **train only**; log10 on skewed features is a fixed transform |
| ANN | Keras MLP, default `64 ReLU → Dropout 0.2 → 32 ReLU → softmax(K)`, Adam 1e-3, sparse CE, batch 32, 100 epochs, early stopping (patience 12, restore best), fixed seeds; every layer configurable |
| Baselines | Logistic Regression (scaled), Random Forest, Decision Tree, XGBoost (optional) – trained on the **same cohorts/weights** as each ANN experiment |

### The three experiments

* **A – Baseline:** all eligible rows, no score.
* **B – High-confidence filtered:** keep CANDIDATE if `koi_score ≥ t`, FALSE POSITIVE if `koi_score ≤ 1 − t` (default t = 0.80; sweep 0.70/0.80/0.90).
  Trained on a **smaller** population (always reported) and scored on the *same original test set* **and** on the high-confidence part of it.
* **C – Confidence-weighted:** `sample_weight` = score (Candidate) or 1 − score (False Positive), floor `min_weight`,
  rescaled to training mean 1; missing scores excluded by default (or neutral weight). Training only; ordinary test metrics.
  A proposed heuristic to be **tested, not assumed to help**.

### Evaluation
Accuracy, balanced accuracy, precision/recall/macro-F1, per-class report, confusion matrix, log loss, Brier score
(binary: positive-class; multiclass: summed, range 0–2), **top-label ECE (10 equal-width bins)**, reliability diagram,
ROC/PR curves, ANN learning curves, best epoch and overfitting notes. Accuracy alone is explicitly shown to be insufficient (class imbalance, calibration).

### Population analysis (observed catalogue only)
Radius, period, period-vs-radius, host-star trends under (1) all records, (2) high-confidence filtered, (3) confidence-weighted.
Weighted mean/SD/median/IQR with Kish effective n, percentile **bootstrap CIs**, **paired-bootstrap** differences (strategies are nested),
weighted KS statistic (descriptive; p-values approximate), Cohen's *d* on log10 values with conventional labels, Candidate-vs-FP group tests.
ANN-prediction summaries use **held-out rows only** and are labelled exploratory.

### Robustness
Multi-seed mean ± SD with **paired stability verdicts**, threshold sweep (common reference test subset), weight-setting sweep,
grouped vs stratified split, leave-one-feature-out, class-distribution comparison, population-summary stability, optional second-dataset comparison.

## 5. Outputs

* Downloads on **Results Report**: CSV tables (individual/ZIP), PNG graphs (individual/ZIP), Markdown research summary, saved model files.
* **Save models + preprocessor** (ANN Workbench) writes `models/saved_models/run_<timestamp>/` with `*.keras`, `*.joblib`,
  `preprocessor.joblib` (fitted imputer + scaler), feature names and `experiment_config.json`.
* "Write everything to results/" fills `results/metrics`, `results/graphs`, `results/reports`.

## 6. Interface

Black-and-grey space theme (near-black base, cool grey surfaces, one steel-blue accent), an animated starfield, a hero with revolving planets on the Overview page, consistent page headers, a session-progress panel in the sidebar and dark-themed charts whose series colours were validated for colour-blind separation and contrast on the dark surface. Theme tokens live in `.streamlit/config.toml` and `src/app_helpers.py`; add your name in `AUTHOR_LINE` there to show it in the sidebar footer.

## 7. Project layout

```
ExoBias.py                    Overview (entry page)
scripts/make_space_images.py  Renders the illustrative planet / nebula / transit images in assets/images/ (artistic renders, not telescope photos)
pages/1..7_*.py
src/data_loader.py           NASA-CSV loading        src/population_analysis.py  weighted stats, bootstrap, KS, effect sizes
src/data_validation.py       class/feature checks,   src/uncertainty_analysis.py robustness suite
                             leakage guard           src/visualizations.py       Matplotlib figures (dark theme)
src/preprocessing.py         clean, split, scale     src/report_generator.py     research summary
src/ann_model.py             Keras ANN               src/app_helpers.py          Streamlit glue only
src/baseline_models.py       sklearn/XGBoost
src/experiments.py           Experiments A/B/C
src/evaluation.py            metrics, ECE, curves
tests/                       70 pytest tests
```

## 8. Limitations (state these in your viva)

* Catalogue labels are **not ground truth**; models measure *agreement with dispositions*, not discovery of planets.
* `koi_score` is **not a calibrated probability**; B and C are sensitivity analyses.
* The KOI catalogue is **detection-selected**: distribution differences are not intrinsic population differences; no occurrence rates are estimated.
* ANN predictions are **not astronomical confirmation**.
* B trains on fewer rows than A, so any difference mixes label quality with data quantity.
* KS p-values between nested samples are approximate; use the statistic and paired bootstrap CIs.
* Single-run differences of a few thousandths are noise – use the multi-seed analysis before concluding anything.

## 9. Verification status

Automated tests (`pytest`) cover validation, leakage-safe preprocessing, grouped splits, cohorts/weights, metrics (ECE/Brier),
baseline experiments, population statistics, robustness and every figure builder. Tests use **synthetic KOI-shaped rows only as
test scaffolding** (never presented as NASA data). The ANN-training test auto-skips if TensorFlow is missing and runs when it is installed.
