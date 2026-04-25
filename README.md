# Berkeley Circle Current ML

Tidal models capture the predictable, repeating component of currents well, but real conditions in SF Bay regularly deviate from the prediction. This project asks whether ML, using tidal harmonics as a foundation, can learn those deviations from open-source data and meaningfully improve forecast accuracy.

The harder problem turned out to be data, not modeling. Most locations have no usable long-term current record. NOAA's point current stations are sparse and recent; HF Radar provides continuous historical coverage but only at ~6km resolution, anchoring the model to wherever a clean HFR pixel exists. Berkeley Circle (~37.856°N, 122.364°W) was chosen because it has the longest uninterrupted HFR record in the Bay (~88% valid since 2015) and sits at a dynamically interesting location. 

Given a viable observation record, the ML layer works well. An equal-weight ensemble of XGBoost and LSTM achieves a skill score of 0.762 on the 2025 test set — reducing prediction error by 76% relative to harmonics alone — with a direction MAE of 19.4°. The dominant drivers, confirmed by SHAP, are the autoregressive subtidal signal, future tidal rate of change, and Sacramento River discharge.

All experimentation is tracked in MLflow, with hyperparameters, metrics, model artifacts, and SHAP plots logged per run.

---

## Approach

The model predicts the non-tidal residual rather than the full current. Tidal harmonics (via utide) explain the dominant variance; the ML layer focuses capacity on the wind, discharge, and pressure-driven signal that harmonics miss. This uses tidal harmonics as a foundation rather than full hydrodynamic simulation — the approach is deliberately lightweight and data-driven.

Predictions are made in along-channel / cross-channel coordinates derived from empirical PCA rotation of the observed u/v current components. The full current at T+24 is then reconstructed as:

```
full_current = model_residual + utide_tidal_prediction
```

Key modeling decisions:
- **XGBoost** with separate regressors for along- and cross-channel residuals; quantile regression (10th/50th/90th percentile) for uncertainty bands
- **LSTM** (PyTorch, 168h lookback) on the same feature set for sequence-based comparison
- **Ensemble** — equal-weight average of XGBoost and LSTM; best overall performance on all four metrics
- **Expanding-window cross-validation** across 5 folds (2020–2024), with utide coefficients and PCA rotation independently refit on each fold's training window to prevent leakage
- **SHAP** (TreeExplainer) for feature attribution

---

## Results

Test set performance (2025, January–June):

| Model | Skill (speed) | Skill (along) | Skill (cross) | Direction MAE |
|---|---|---|---|---|
| **Ensemble (XGBoost + LSTM)** | **0.762** | **0.703** | **0.407** | **19.4°** |
| LSTM (168h lookback) | 0.758 | 0.700 | 0.392 | 20.0° |
| XGBoost | 0.734 | 0.655 | 0.373 | 21.5° |
| Tidal baseline | 0.000 | 0.000 | 0.000 | — |

Skill score = `1 − (MAE_model / MAE_tidal_only)`. All models substantially outperform the harmonic baseline. The ensemble leads on every metric. Along-channel skill is higher than cross-channel, consistent with the physical expectation that ebb/flood enhancement is more predictable than lateral variability.

---

## Repo structure

```
berkeley-circle-ml/
├── src/
│   ├── config.py              # stations, coordinates, date ranges, hyperparameters
│   ├── run_etl.py             # fetch all data sources, build training dataset
│   ├── train.py               # CV + XGBoost + LSTM training, MLflow logging
│   ├── predict.py             # 24h-ahead inference on historical data
│   ├── etl/
│   │   ├── fetch.py           # one function per data source
│   │   └── features.py        # utide fitting, PCA rotation, feature engineering
│   ├── models/
│   │   ├── xgboost_model.py   # point + quantile XGBoost, CV loop
│   │   ├── lstm_model.py      # PyTorch LSTM, training + inference
│   │   └── baseline.py        # tidal-only baseline, skill score
│   └── evaluation/
│       ├── metrics.py         # MAE, skill score, direction MAE
│       └── plots.py           # time series, scatter, SHAP, CV skill plots
├── notebooks/
│   └── map_predictions.ipynb  # load best model, generate predictions, overlay on SF Bay map
├── tests/                     # unit tests for ETL and feature engineering
├── data/                      # gitignored — regenerate with ETL (see below)
└── mlruns/                    # gitignored — MLflow run artifacts
```

---

## Quickstart

**1. Create environment**

```bash
conda env create -f environment.yml
conda activate berkeley-circle-ml
pip install -e .
```

**2. Fetch data and build dataset**

```bash
python src/run_etl.py
```

Fetches all data sources and writes the feature-engineered training CSV to `data/`. The HFR current fetch takes ~45 minutes on first run; subsequent runs load from a local parquet cache.

**3. Train models**

```bash
python src/train.py
```

Runs 5-fold expanding-window CV, trains final XGBoost and LSTM models, and logs all hyperparameters, metrics, model artifacts, and SHAP plots to MLflow.

**4. Run inference on a historical date**

```bash
python src/predict.py "2025-04-01 12:00"
```

Outputs the 24h-ahead current forecast for a given issue time using the trained models. See note below on inference limitations.

**5. Explore predictions on a map**

```bash
jupyter notebook notebooks/map_predictions.ipynb
```

Loads the best XGBoost model from MLflow, generates predictions over the 2025 test set, and produces five figures overlaid on a Cartopy map of SF Bay: a context map showing the Berkeley Circle location and predictor stations, a quiver fan of hourly current vectors (observed vs predicted) for a two-week window, monthly panels showing mean predicted current and MAE, a directional current rose comparing observed and predicted distributions, and a speed scatter/bias plot.

> **Note:** The notebook uses XGBoost-only predictions due to local jupyter kernel limitations for mlflow.pytorch. Scripts are unaffected and use the full ensemble.

**6. Explore results in MLflow**

```bash
mlflow ui --port 5001 --backend-store-uri mlruns
```

Open http://localhost:5001 to browse runs, compare metrics, and view logged plots (time series, scatter, SHAP beeswarm) directly in the MLflow UI.

**7. Run tests**

```bash
pytest tests/
```

---

## Inference limitations

`src/predict.py` operates on the historical feature dataset and is an exercise in applying the trained models rather than a deployable forecasting tool. True real-time inference is not practical with this data stack for two reasons:

1. **HFR data lag.** The NCEI HFR archive that provides observed surface currents has a delay exceeding 48 hours. The autoregressive current features (`resid_along`, `resid_cross`) are the top SHAP drivers — without recent observations, these features are unavailable and forecast quality degrades significantly.

2. **Wind forcing.** Training used ERA5 reanalysis (historical). A real-time forecast would require an NWP wind product at T+24, which is available but not wired up here.

A production version would replace the NCEI HFR source with a real-time feed (e.g., CORDC near-real-time API) and integrate Open-Meteo's forecast endpoint for T+24 wind.

---

## Data sources

| Source | Product | Station / API |
|---|---|---|
| NOAA CO-OPS | Water level, tide predictions | Richmond 9414863, Fort Point 9414290, Point Reyes 9415020 |
| NOAA/UCSD HFR USWC 6km | Surface currents (u, v) | NCEI archive, ~37.856°N 122.364°W |
| USGS NWIS | Sacramento River discharge | Site 11447650 (Freeport) |
| CDEC | Net Delta outflow | Station DTO, Sensor 23 |
| Open-Meteo / ERA5 | Wind (u, v) | 37.86°N, 122.40°W |
| NOAA PFEL | Coastal upwelling index | 36°N, 122°W |

No data files are committed. See `src/config.py` for station IDs and date ranges.
