"""
XGBoost models for along-channel and cross-channel residual prediction.

Includes:
  - Point estimate model (mean prediction)
  - Quantile regression (10th / 50th / 90th percentile)
  - Expanding-window cross-validation with per-fold utide/PCA refitting

CV leakage fixes applied here:
  - utide coefficients refit on each fold's training window
  - PCA rotation refit on each fold's training window
  - XGBoost handles NaN natively — no fillna(0)
"""

import logging

import mlflow
import mlflow.xgboost
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error

from src.config import (
    CV_FOLDS,
    FORECAST_HOURS,
    QUANTILES,
    XGB_PARAMS,
)
from src.etl.features import (
    _add_harmonic_features,
    fit_pca_rotation,
    fit_utide,
    get_feature_cols,
    rotate_currents,
)
from src.models.baseline import skill_score, tidal_baseline_mae

logger = logging.getLogger(__name__)

TARGETS = ["resid_along_fwd", "resid_cross_fwd"]


# ── Training ──────────────────────────────────────────────────────────────────

def train_xgboost(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list[str],
) -> dict[str, xgb.XGBRegressor]:
    """Fit separate XGBoost regressors for along and cross residuals.

    Uses val_df for early stopping only — not reported as test performance.
    XGBoost handles NaN natively; no imputation applied.

    Returns dict mapping target name → fitted model.
    """
    models = {}
    for target in TARGETS:
        y_tr = train_df[target].dropna()
        X_tr = train_df.loc[y_tr.index, feature_cols]

        y_va = val_df[target].dropna()
        X_va = val_df.loc[y_va.index, feature_cols]

        model = xgb.XGBRegressor(**XGB_PARAMS)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        models[target] = model
        logger.info("XGBoost %s: best iteration %d", target, model.best_iteration)

    return models


def train_quantile_xgboost(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list[str],
    quantiles: list[float] = QUANTILES,
) -> dict[tuple[str, float], xgb.XGBRegressor]:
    """Fit quantile XGBoost regressors for calibrated uncertainty bands.

    Returns dict mapping (target, quantile) → fitted model.
    """
    # early_stopping_rounds and eval_metric are incompatible with reg:quantileerror
    q_params = {k: v for k, v in XGB_PARAMS.items()
                if k not in ("eval_metric", "early_stopping_rounds")}
    q_params["objective"] = "reg:quantileerror"

    models = {}
    for target in TARGETS:
        y_tr = train_df[target].dropna()
        X_tr = train_df.loc[y_tr.index, feature_cols]

        for q in quantiles:
            model = xgb.XGBRegressor(**q_params, quantile_alpha=q)
            model.fit(X_tr, y_tr, verbose=False)
            models[(target, q)] = model
            logger.info("XGBoost quantile %s q=%.1f: done", target, q)

    return models


# ── Cross-validation ──────────────────────────────────────────────────────────

def _build_fold_features(raw_df: pd.DataFrame,
                         pca_angle: float, utide_coef: dict,
                         forecast_hours: int) -> pd.DataFrame:
    """Add rotated residuals and future targets to a fold DataFrame.

    Called after per-fold utide/PCA refitting so features reflect
    fold-specific rotation — avoids CV leakage.
    """
    fh = forecast_hours
    df = raw_df.copy()

    from src.etl.features import reconstruct_tidal_currents
    tidal = reconstruct_tidal_currents(df.index, utide_coef)
    df["tide_curr_u"] = tidal["tide_curr_u"]
    df["tide_curr_v"] = tidal["tide_curr_v"]

    curr_along, curr_cross = rotate_currents(df["curr_u"], df["curr_v"], pca_angle)
    tide_along, tide_cross = rotate_currents(df["tide_curr_u"], df["tide_curr_v"], pca_angle)

    df["resid_along"] = curr_along - tide_along
    df["resid_cross"] = curr_cross - tide_cross

    # Future tidal currents
    if f"tide_curr_u_{fh}h" in df.columns:
        ta24, tc24 = rotate_currents(
            df[f"tide_curr_u_{fh}h"], df[f"tide_curr_v_{fh}h"], pca_angle
        )
        df[f"tide_curr_along_{fh}h"] = ta24
        df[f"tide_curr_cross_{fh}h"] = tc24

    df["resid_along_fwd"] = df["resid_along"].shift(-fh)
    df["resid_cross_fwd"] = df["resid_cross"].shift(-fh)

    df = _add_harmonic_features(df)

    # Drop redundant raw columns — consistent with engineer_features()
    drop_cols = ["obs_wl", "discharge_cms", "outflow_cms",
                 "tide_curr_u", "tide_curr_v", "curr_u", "curr_v"]
    return df.drop(columns=[c for c in drop_cols if c in df.columns])


def run_cv(raw_df: pd.DataFrame,
           folds: list[tuple] = CV_FOLDS,
           forecast_hours: int = FORECAST_HOURS) -> pd.DataFrame:
    """Expanding-window cross-validation for XGBoost.

    Per-fold: refits utide coefficients and PCA rotation on the fold's
    training window before building features. This prevents leakage of
    future tidal/rotation information into earlier folds.

    Logs each fold's metrics to the active MLflow run.

    Returns a DataFrame of fold-level metrics.
    """
    results = []

    for fold_idx, (train_s, train_e, val_s, val_e) in enumerate(folds):
        logger.info("CV fold %d: train %s–%s, val %s–%s",
                    fold_idx + 1, train_s, train_e, val_s, val_e)

        fold_train_raw = raw_df.loc[train_s:train_e]
        fold_val_raw   = raw_df.loc[val_s:val_e]

        # Refit utide and PCA on this fold's training window
        tr_mask = fold_train_raw["curr_u"].notna() & fold_train_raw["curr_v"].notna()
        utide_coef = fit_utide(
            fold_train_raw.loc[tr_mask, "curr_u"],
            fold_train_raw.loc[tr_mask, "curr_v"],
        )
        pca_angle = fit_pca_rotation(
            fold_train_raw.loc[tr_mask, "curr_u"],
            fold_train_raw.loc[tr_mask, "curr_v"],
        )

        fold_train = _build_fold_features(fold_train_raw, pca_angle, utide_coef, forecast_hours)
        fold_val   = _build_fold_features(fold_val_raw,   pca_angle, utide_coef, forecast_hours)

        feature_cols = get_feature_cols(fold_train, forecast_hours)

        # Val split within fold for early stopping (last year of training window)
        val_year = int(train_e[:4])
        xgb_tr = fold_train[fold_train.index.year < val_year]
        xgb_va = fold_train[fold_train.index.year >= val_year]

        models = train_xgboost(xgb_tr, xgb_va, feature_cols)

        fold_row = {"val_year": val_s[:4]}
        for target in TARGETS:
            component = target.replace("resid_", "").replace("_fwd", "")
            y_va = fold_val[target].dropna()
            X_va = fold_val.loc[y_va.index, feature_cols]
            preds = models[target].predict(X_va)

            mae_ml   = mean_absolute_error(y_va, preds)
            mae_tide = tidal_baseline_mae(y_va)
            skill    = skill_score(mae_ml, mae_tide)

            fold_row[f"mae_{component}"]   = mae_ml
            fold_row[f"skill_{component}"] = skill

            mlflow.log_metrics({
                f"cv_fold{fold_idx+1}_mae_{component}":   mae_ml,
                f"cv_fold{fold_idx+1}_skill_{component}": skill,
            })

        results.append(fold_row)
        logger.info("  Fold %d: skill_along=%.3f  skill_cross=%.3f",
                    fold_idx + 1, fold_row.get("skill_along", float("nan")),
                    fold_row.get("skill_cross", float("nan")))

    cv_df = pd.DataFrame(results).set_index("val_year")
    mlflow.log_metrics({
        "cv_mean_skill_along": cv_df["skill_along"].mean(),
        "cv_mean_skill_cross": cv_df["skill_cross"].mean(),
    })
    return cv_df
