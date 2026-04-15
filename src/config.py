"""
Central configuration for richmond-current-ml.
All constants live here — stations, coordinates, date ranges, model parameters.
"""

# ── Date range ─────────────────────────────────────────────────────────────────
BEGIN_ISO = "2015-01-01"
END_ISO   = "2025-12-31"
TRAIN_END = "2024-12-31 23:00"
TEST_START = "2025-01-01"

# ── Forecast horizon ───────────────────────────────────────────────────────────
FORECAST_HOURS = 24

# ── HFR — Richmond Reach pixel ────────────────────────────────────────────────
HFR_LAT = 37.856
HFR_LON = -122.364
HFR_SEARCH_RADIUS_DEG = 0.1   # nearest-valid pixel search half-width

# ── CO-OPS stations ───────────────────────────────────────────────────────────
COOPS_RICHMOND   = "9414863"   # Richmond — WL, tide predictions, SST
COOPS_FORTPOINT  = "9414290"   # Fort Point — WL, pressure
COOPS_POINTREYES = "9415020"   # Point Reyes — WL (remote ocean forcing)

COOPS_API = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"

# ── USGS discharge ─────────────────────────────────────────────────────────────
USGS_SAC_SITE = "11447650"     # Sacramento River at Freeport (nearest to Bay)
CFS_TO_CMS    = 0.0283168      # cubic feet/s → cubic metres/s

# ── CDEC delta outflow ────────────────────────────────────────────────────────
CDEC_OUTFLOW_STA    = "DTO"
CDEC_OUTFLOW_SENSOR = "23"
CDEC_API = "https://cdec.water.ca.gov/dynamicapp/req/CSVDataServlet"

# ── Wind — ERA5 via Open-Meteo ────────────────────────────────────────────────
OPENMETEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
WIND_LAT = 37.86
WIND_LON = -122.40

# ── Upwelling index — NOAA PFEL ───────────────────────────────────────────────
PFEL_URL = (
    "https://coastwatch.pfeg.noaa.gov/erddap/tabledap/erdUI366hr.csv"
    "?time,upwelling_index&time>={begin}&time<={end}"
)

# ── HFR cache ─────────────────────────────────────────────────────────────────
HFR_CACHE_PATH = "data/hfr_cache_2015_2025.parquet"
TRAIN_CSV_PATH = "data/richmond_reach_train_{begin}_{end}.csv"

# ── Tidal constituents (angular frequency, radians/hour) ──────────────────────
TIDAL_CONSTITUENTS = {
    "M2": 0.5059, "S2": 0.5236, "K1": 0.2625,
    "O1": 0.2434, "N2": 0.4963, "M4": 1.0118,
}

# ── Features ──────────────────────────────────────────────────────────────────
OBS_FEATURES = [
    "ntr", "dtide_dt",
    "ntr_lag6h", "ntr_lag12h", "ntr_lag24h",
    "ntr_gradient", "ntr_pointreyes", "ntr_fortpoint",
    "resid_along", "resid_cross",
    "pressure_hpa",
    "discharge_lag1d", "discharge_lag2d", "discharge_lag3d",
    "outflow_lag1d", "outflow_lag2d",
    "upwelling_idx",
    "hour_sin", "hour_cos", "doy_sin", "doy_cos",
]

FUT_FEATURE_ROOTS = [
    "pred_tide", "dtide_dt",
    "tide_curr_u", "tide_curr_v",   # rotated to along/cross inside CV
    "wind_u", "wind_v", "wind_spd_max_d",
    "pressure_hpa",
]

# cos_M2 retained — SHAP ranks it above dtide_dt_24h in cross-channel
# All other harmonics removed: redundant given tide_curr_u/v_24h from utide
HARMONIC_REMOVE = {
    "cos_N2", "sin_N2",
    "cos_K1", "sin_K1",
    "cos_O1", "sin_O1",
    "cos_M4",
    "cos_S2", "sin_S2",
    "sin_M2",
}

TARGET_COLS = ["resid_along_fwd", "resid_cross_fwd"]

# ── Model hyperparameters ─────────────────────────────────────────────────────
XGB_PARAMS = dict(
    n_estimators=2000,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    reg_lambda=2,
    early_stopping_rounds=50,
    eval_metric="rmse",
    random_state=42,
    n_jobs=-1,
)

LSTM_LOOKBACK  = 168    # hours (7 days); testing vs 48h — tidal memory may be captured by utide features
LSTM_HIDDEN    = 64
LSTM_LAYERS    = 1
LSTM_DROPOUT   = 0.1
LSTM_EPOCHS    = 100
LSTM_PATIENCE  = 15
LSTM_BATCH     = 256
LSTM_VAL_YEAR  = 2024   # carved from train for early stopping; not reported as test

QUANTILES = [0.1, 0.5, 0.9]

# ── Cross-validation folds ────────────────────────────────────────────────────
CV_FOLDS = [
    ("2015-01-01", "2019-12-31 23:00", "2020-01-01", "2020-12-31 23:00"),
    ("2015-01-01", "2020-12-31 23:00", "2021-01-01", "2021-12-31 23:00"),
    ("2015-01-01", "2021-12-31 23:00", "2022-01-01", "2022-12-31 23:00"),
    ("2015-01-01", "2022-12-31 23:00", "2023-01-01", "2023-12-31 23:00"),
    ("2015-01-01", "2023-12-31 23:00", "2024-01-01", "2024-12-31 23:00"),
]

# ── Evaluation ────────────────────────────────────────────────────────────────
SLACK_SPEED_MS = 0.05   # NOAA definition; direction MAE filtered for speed > this
SPEED_CLIP_MS  = 1.5    # physical maximum at Richmond Reach

# ── MLflow ────────────────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI = "mlruns"
MLFLOW_EXPERIMENT   = "richmond-current-24h"
MLFLOW_RUN_NAME     = "xgboost+lstm_168h_dropout01_noharmonics"

# ── Paths ─────────────────────────────────────────────────────────────────────
ARTIFACTS_DIR = "data/artifacts"

# ── Plot settings ─────────────────────────────────────────────────────────────
PLOT_WINDOW_START = "2025-02-01"
PLOT_WINDOW_END   = "2025-02-14"
