# 🚦 Sensorless Traffic — predicting urban congestion where the city has no sensors

> **EDM course project.** An interactive Streamlit application that turns the
> tiny fraction of streets a city actually *measures* into a **city-wide
> congestion map**, and transfers that map to a neighbouring town that has **no
> traffic sensors at all**.

| | |
|---|---|
| **Live app** | _add your Streamlit Cloud URL here_ |
| **Demo video** | _add your ≤5 min video link here_ |
| **Data** | [Valencia Open Data](https://valencia.opendatasoft.com/) (loop sensors + real-time traffic state) · [OpenStreetMap](https://www.openstreetmap.org/) |
| **Stack** | Python · scikit-learn · XGBoost · Streamlit · Folium (Leaflet) |

---

## The city problem

A municipality can only **measure** traffic where it has installed sensors.
In Valencia, electromagnetic loop sensors cover barely **2.4 %** of street
segments — the other **97.6 % are blind spots**. Smaller towns (e.g. Paiporta)
often have **none**. Mobility decisions — where to add a bus stop, whether to
narrow a street, which junctions to redesign — are therefore made with almost no
data.

**Our idea:** congestion is largely written into a street's *physical shape*
(road class, number of lanes, speed limit, length, structure). So we learn the
mapping `road shape → congestion` on the few sensored streets and **predict it
everywhere** — then transfer it to a sensorless town.

## What the app does (4 tabs)

1. **🗺️ Congestion map** — the full Valencia network (54k segments) rendered on a
   interactive Leaflet map. Toggle between the *model prediction for the whole city* and
   *sensors only*, to literally see the blind-spot problem. Filter by road class
   and congestion level.
2. **🧪 Scenario simulator** — pick a real street (or design one), change its
   lanes / speed limit / road class / direction, and watch the model re-estimate
   its congestion live. A decision-support tool for *before* you build.
3. **🌍 Transfer · Paiporta** — the Valencia model + its learned per-road-class
   priors produce a complete congestion map for Paiporta, built purely from
   OpenStreetMap geometry, with **zero local sensors**.
4. **📊 Model & method** — honest held-out + cross-validated metrics, feature
   importance, confusion matrix, the end-to-end pipeline, a **live feed** from
   Valencia's real-time traffic dataset, and limitations.

## Data-science methodology

The application is the productisation of a full DS pipeline (originally explored
in the notebooks under `../notebooks/`, refactored here into `src/`):

| Stage | What we do | Methods |
|---|---|---|
| **Acquire** | Valencia drive network from OSMnx; loop-sensor hourly intensity from Valencia Open Data | OSM graph extraction |
| **Clean & impute** | Median imputation of `lanes` per road class; **spatial kNN** imputation of `maxspeed`; missingness kept as features; structural tags binarised | EDA, kNN imputation, missingness analysis |
| **Spatial join** | A custom Haversine *point-on-segment* test assigns each sensor to its street segment | Computational geometry, `O(N·E)` |
| **Target** | Hourly vehicle counts bucketed into 4 quartile classes (low/medium/high/peak) | Quantile binning |
| **Validate signal** | Contingency tables, one-way **ANOVA** + **Tukey HSD** confirm speed/road-class discriminate congestion | Inferential statistics |
| **Model** | Baseline → Decision Tree → Random Forest → MLP (GridSearchCV) → **XGBoost** | Model selection, 5-fold CV, learning curves |
| **Transfer** | Per-road-class priors learned in Valencia clean & impute a sensorless town | Domain transfer |

### Results (honest)

Trained on **1,292** sensor-labelled segments, evaluated on a stratified 30 %
hold-out:

| Metric | Value |
|---|---|
| Test F1-macro | **0.53** |
| Balanced accuracy | **0.56** |
| Cohen's κ | **0.39** |
| 5-fold CV F1-macro | **0.52 ± 0.03** |

A majority-class baseline scores ~0.10 F1-macro, so the model roughly
**quintuples** it using *only static road attributes* — no historical or
real-time flow. κ ≈ 0.39 reflects a genuinely hard, data-scarce problem rather
than an over-fitted demo. Most confusion is between *adjacent* levels.

## Run it locally

```bash
# from the repository root, with the project venv (Python 3.12)
uv sync                      # or: pip install -r EDM/requirements.txt
.venv/bin/streamlit run EDM/app/app.py
```

### Rebuild the artifacts from raw data (optional)

```bash
.venv/bin/python EDM/scripts/build_artifacts.py
```

This re-trains the model, scores the whole Valencia network, and downloads +
scores Paiporta, writing `EDM/artifacts/*` and `EDM/data/*.parquet`.

## Deploy on Streamlit Community Cloud

1. Push this repository to GitHub.
2. On [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. **Main file path:** `EDM/app/app.py` · **Python version:** 3.12.
4. Dependencies are read from the root `requirements.txt` (runtime-only — no
   OSMnx/GeoPandas, so the build is fast). The live tab reaches Valencia Open
   Data automatically; offline it falls back to the bundled snapshot.

## Project structure

```
EDM/
├── app/app.py                  # Streamlit application (4 tabs)
├── src/
│   ├── pipeline.py             # OSM cleaning, imputation, feature contract
│   ├── modeling.py             # XGBoost bundle: train / evaluate / predict / persist
│   └── io_live.py              # real-time Valencia traffic feed (+ fallback)
├── scripts/build_artifacts.py  # offline: build model + scored maps
├── artifacts/                  # model.joblib + metrics.json
├── data/                       # scored segments (parquet) + snapshot
└── requirements.txt
```

## Team

EDM team — **Imad Rifai · Raúl · Nouh**.

## How this project maps to the EDM syllabus

The project is a deliberate application of the course topics:

- **Topic 1 · Model evaluation.** The *Model & validation* tab reports the
  threshold/qualitative metrics from Topic 1 for a 4-class classifier:
  **confusion matrix, balanced accuracy, F1-macro and Cohen's κ**, plus 5-fold
  cross-validation. We deliberately avoid plain accuracy because the classes are
  balanced by construction (quartiles) but the problem is hard.
- **Topic 2 · Ensemble methods.** Model selection is a Topic-2 comparison:
  a single **decision tree** → **bagging (Random Forest)** → **boosting
  (XGBoost)**. Boosting wins, consistent with the bias–variance trade-off
  (Topic 1) — lower variance than a single deep tree, lower bias than bagging here.
- **Topic 4 · Structured/automated DS (CRISP-DM).** The pipeline follows the
  CRISP-DM phases: problem analysis → data preparation & cleaning (*data
  wrangling*: imputation, binarisation, missingness-as-feature) → exploration &
  feature engineering → modelling → deployment.
- **Topic 5 · Deployment.** The app is a **server-side deployment** (Streamlit
  Cloud) that combines **batch** prediction (the whole network is scored offline
  into parquet) with a **streaming/API** live feed (real-time traffic, ~3 min) —
  the hybrid pattern described in Topic 5.
- **Topic 3 · Fairness & causality (lightly).** The scenario simulator is a
  *what-if intervention* — conceptually `do(lanes = k)`. We state honestly that it
  shows the model's **associational** response, not a validated causal effect
  (Topic 3's "observing ≠ intervening").
- **Topic 6 · Monitoring (future work).** The live feed is the natural hook for
  **data/feature-drift** monitoring; training on a single snapshot is itself a
  drift-related limitation.

## Academic note

The data-science engine reuses the authors' own prior university-internship work
on Valencia/Paiporta mobility (notebooks in `../notebooks/`, in Spanish). **All
EDM deliverables here are new**: the interactive application, the spatial-transfer
formulation, the live-data integration, the scenario simulator, and the
deployment.

## Limitations & future work

- The model learns from a **single one-hour snapshot**, so it captures the
  *spatial* structure of congestion, not time-of-day dynamics. Ingesting
  historical/real-time series is the natural next step.
- Transfer to Paiporta assumes its streets behave like Valencia streets of the
  same class — reasonable for a neighbouring town, to be validated once sensors
  exist.
