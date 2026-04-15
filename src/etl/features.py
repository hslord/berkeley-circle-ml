"""
Feature engineering: merges raw data sources into a single hourly DataFrame,
fits utide tidal constituents, performs PCA rotation to along/cross-channel
coordinates, and engineers all model features.

Key design decisions:
  - utide is fit on training data only (≤ TRAIN_END) to avoid leakage
  - PCA rotation is fit on training data only
  - Both utide coefficients and PCA angle are returned for use in CV
    (callers must refit per fold — do not reuse the full-period fits in CV)
  - Null handling per physical timescale of each variable
"""

import logging

import numpy as np
import pandas as pd
import utide
from sklearn.decomposition import PCA

from src.config import (
    BEGIN_ISO,
    END_ISO,
    FORECAST_HOURS,
    HARMONIC_REMOVE,
    OBS_FEATURES,
    TIDAL_CONSTITUENTS,
    TRAIN_END,
)

logger = logging.getLogger(__name__)


# ── Merge ─────────────────────────────────────────────────────────────────────

def merge_sources(
    obs_wl_richmond: pd.Series,
    pred_tide_richmond: pd.Series,
    obs_wl_fortpoint: pd.Series,
    pred_tide_fortpoint: pd.Series,
    obs_wl_pointreyes: pd.Series,
    pred_tide_pointreyes: pd.Series,
    pressure: pd.Series,
    sst: pd.Series,
    hfr: pd.DataFrame,
    discharge: pd.Series,
    outflow: pd.Series,
    wind: pd.DataFrame,
    upwelling: pd.Series,
    begin_iso: str = BEGIN_ISO,
    end_iso: str = END_ISO,
) -> pd.DataFrame:
    """Merge all sources onto a complete hourly index.

    Returns a single DataFrame with all raw columns before feature engineering.
    Missing values are interpolated according to the physical timescale of
    each variable (see _fill_nulls).
    """
    idx = pd.date_range(begin_iso, end_iso, freq="h")
    df  = pd.DataFrame(index=idx)
    df.index.name = None

    # Water level & tidal features — Richmond
    df["obs_wl"]    = obs_wl_richmond.reindex(idx)
    df["pred_tide"] = pred_tide_richmond.reindex(idx)
    df["ntr"]       = (df["obs_wl"] - df["pred_tide"]).rename("ntr")
    df["dtide_dt"]  = df["pred_tide"].diff().rename("dtide_dt")

    # Fort Point
    ntr_fp          = (obs_wl_fortpoint.reindex(idx) - pred_tide_fortpoint.reindex(idx))
    df["ntr_fortpoint"] = ntr_fp

    # Point Reyes
    ntr_pr              = (obs_wl_pointreyes.reindex(idx) - pred_tide_pointreyes.reindex(idx))
    df["ntr_pointreyes"] = ntr_pr

    # Barotropic pressure gradient
    df["ntr_gradient"] = df["ntr"] - df["ntr_fortpoint"]

    # Pressure & SST
    df["pressure_hpa"] = pressure.reindex(idx)
    df["sst_c"]        = sst.reindex(idx)

    # HFR currents
    df["curr_u"] = hfr["curr_u"].reindex(idx)
    df["curr_v"] = hfr["curr_v"].reindex(idx)

    # Discharge
    df["discharge_cms"] = discharge.reindex(idx)

    # Delta outflow
    df["outflow_cms"] = outflow.reindex(idx)

    # Wind
    for col in ["wind_u", "wind_v", "wind_spd_max_d"]:
        df[col] = wind[col].reindex(idx)

    # Upwelling (daily → forward-fill to hourly)
    df["upwelling_idx"] = upwelling.resample("h").ffill().reindex(idx)

    return _fill_nulls(df)


def _fill_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """Interpolate missing values per physical timescale.

    Targets (curr_u, curr_v) are NOT filled — NaN rows are dropped at train time.
    """
    tidal_cols  = ["obs_wl", "pred_tide", "ntr", "dtide_dt",
                   "ntr_fortpoint", "ntr_pointreyes", "ntr_gradient"]
    wind_cols   = ["wind_u", "wind_v", "wind_spd_max_d"]
    pres_cols   = ["pressure_hpa"]
    river_cols  = ["discharge_cms", "outflow_cms"]
    sst_col     = ["sst_c"]
    upwell_col  = ["upwelling_idx"]

    df[tidal_cols]  = df[tidal_cols].interpolate(method="time", limit=6)
    df[wind_cols]   = df[wind_cols].interpolate(method="time", limit=6)
    df[pres_cols]   = df[pres_cols].interpolate(method="time", limit=6)
    df[river_cols]  = df[river_cols].interpolate(method="time", limit=48)
    df[sst_col]     = df[sst_col].interpolate(method="time", limit=48)
    df[upwell_col]  = df[upwell_col].ffill(limit=48)

    return df


# ── Tidal harmonic analysis ───────────────────────────────────────────────────

def fit_utide(curr_u: pd.Series, curr_v: pd.Series) -> dict:
    """Fit utide tidal constituents on the provided current time series.

    Must be called on training data only. Returns the utide coefficient object.
    """
    valid = curr_u.notna() & curr_v.notna()
    t_hours = (curr_u.index[valid] - pd.Timestamp("2000-01-01")).total_seconds() / 3600
    coef = utide.solve(
        t_hours,
        curr_u.values[valid],
        curr_v.values[valid],
        lat=37.86,
        method="ols",
        conf_int="linear",
        verbose=False,
    )
    return coef


def reconstruct_tidal_currents(index: pd.DatetimeIndex, coef: dict) -> pd.DataFrame:
    """Reconstruct tidal u/v from utide coefficients at all times in index."""
    t_hours = (index - pd.Timestamp("2000-01-01")).total_seconds() / 3600
    tide = utide.reconstruct(t_hours, coef, verbose=False)
    return pd.DataFrame(
        {"tide_curr_u": tide.u, "tide_curr_v": tide.v},
        index=index,
    )


# ── PCA rotation to along/cross-channel ──────────────────────────────────────

def fit_pca_rotation(curr_u: pd.Series, curr_v: pd.Series) -> float:
    """Fit PCA on (u, v) and return rotation angle (radians).

    First principal component = along-channel (dominant flow axis).
    Must be called on training data only.
    """
    valid = curr_u.notna() & curr_v.notna()
    uv = np.column_stack([curr_u.values[valid], curr_v.values[valid]])
    pca = PCA(n_components=2).fit(uv)
    angle = float(np.arctan2(pca.components_[0, 1], pca.components_[0, 0]))
    return angle


def rotate_currents(u: pd.Series, v: pd.Series, angle: float) -> tuple[pd.Series, pd.Series]:
    """Rotate (u, v) to (along, cross) channel coordinates."""
    c, s = np.cos(angle), np.sin(angle)
    along = c * u + s * v
    cross = -s * u + c * v
    return along, cross


# ── Feature engineering ───────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame, utide_coef: dict,
                      pca_angle: float) -> pd.DataFrame:
    """Add all derived features to the merged DataFrame.

    Adds:
      - Tidal current reconstruction + residuals in along/cross coordinates
      - Future-shifted features (T+FORECAST_HOURS)
      - Discharge / outflow lags
      - Cyclic time encodings
      - NTR lag features
      - Harmonic tidal features (pending SHAP-driven removal)
      - Forward targets (resid_along_fwd, resid_cross_fwd)

    Does NOT scale or drop NaN rows — that happens in the model layer.
    """
    fh = FORECAST_HOURS
    df = df.copy()

    # Tidal current reconstruction
    tidal = reconstruct_tidal_currents(df.index, utide_coef)
    df["tide_curr_u"] = tidal["tide_curr_u"]
    df["tide_curr_v"] = tidal["tide_curr_v"]

    # Rotate observed and tidal currents to along/cross
    curr_along, curr_cross = rotate_currents(df["curr_u"], df["curr_v"], pca_angle)
    tide_along, tide_cross = rotate_currents(df["tide_curr_u"], df["tide_curr_v"], pca_angle)

    df["curr_along"]  = curr_along
    df["curr_cross"]  = curr_cross
    df["tide_along"]  = tide_along
    df["tide_cross"]  = tide_cross

    # Residuals at T (autoregressive features — valid, not leakage)
    df["resid_along"] = df["curr_along"] - df["tide_along"]
    df["resid_cross"] = df["curr_cross"] - df["tide_cross"]

    # Future-shifted features (shift by -fh to align T+fh values with row T)
    future_roots = ["pred_tide", "dtide_dt", "tide_curr_u", "tide_curr_v",
                    "wind_u", "wind_v", "wind_spd_max_d", "pressure_hpa"]
    for col in future_roots:
        if col in df.columns:
            df[f"{col}_{fh}h"] = df[col].shift(-fh)

    # Rotate future tidal currents to along/cross
    if f"tide_curr_u_{fh}h" in df.columns and f"tide_curr_v_{fh}h" in df.columns:
        ta24, tc24 = rotate_currents(df[f"tide_curr_u_{fh}h"], df[f"tide_curr_v_{fh}h"], pca_angle)
        df[f"tide_curr_along_{fh}h"] = ta24
        df[f"tide_curr_cross_{fh}h"] = tc24

    # Forward targets: residual at T+fh (what the model predicts)
    df["resid_along_fwd"] = df["resid_along"].shift(-fh)
    df["resid_cross_fwd"] = df["resid_cross"].shift(-fh)

    # Discharge lags (Sacramento: 1–3 day transit to Richmond Reach)
    df["discharge_lag1d"] = df["discharge_cms"].shift(24)
    df["discharge_lag2d"] = df["discharge_cms"].shift(48)
    df["discharge_lag3d"] = df["discharge_cms"].shift(72)

    # Delta outflow lags
    df["outflow_lag1d"] = df["outflow_cms"].shift(24)
    df["outflow_lag2d"] = df["outflow_cms"].shift(48)

    # NTR lag features (surge memory / inertia)
    df["ntr_lag6h"]  = df["ntr"].shift(6)
    df["ntr_lag12h"] = df["ntr"].shift(12)
    df["ntr_lag24h"] = df["ntr"].shift(24)

    # Cyclic time encodings
    df["hour_sin"] = np.sin(2 * np.pi * df.index.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df.index.hour / 24)
    df["doy_sin"]  = np.sin(2 * np.pi * df.index.dayofyear / 365.25)
    df["doy_cos"]  = np.cos(2 * np.pi * df.index.dayofyear / 365.25)

    # Harmonic tidal features (pending SHAP-driven removal)
    df = _add_harmonic_features(df)

    # Drop redundant raw columns not used as features.
    # curr_u and curr_v are retained — run_cv needs them to refit utide/PCA per fold.
    drop_cols = ["obs_wl", "discharge_cms", "outflow_cms",
                 "tide_curr_u", "tide_curr_v",
                 "curr_along", "curr_cross", "tide_along", "tide_cross"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    return df


def _add_harmonic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add sin/cos harmonic features for major tidal constituents.

    Some are pre-marked for removal in HARMONIC_REMOVE after SHAP analysis.
    All are included here to allow the experiment to run.
    """
    t_hours = (df.index - pd.Timestamp("2020-01-01")).total_seconds() / 3600
    for name, omega in TIDAL_CONSTITUENTS.items():
        df[f"cos_{name}"] = np.cos(omega * t_hours)
        df[f"sin_{name}"] = np.sin(omega * t_hours)
    return df


def get_feature_cols(df: pd.DataFrame, forecast_hours: int = FORECAST_HOURS) -> list[str]:
    """Return the full ordered feature list for the current DataFrame.

    Excludes target columns, raw unshifted tidal currents, raw curr_u/v
    (retained in df for CV refitting but not model inputs), and pre-marked
    harmonic features.
    """
    fh = forecast_hours
    _exclude = {"curr_u", "curr_v", f"tide_curr_u_{fh}h", f"tide_curr_v_{fh}h"}
    obs = [c for c in OBS_FEATURES if c in df.columns]
    fut = [c for c in df.columns
           if c.endswith(f"_{fh}h")
           and c not in _exclude]
    harmonics = [c for c in df.columns
                 if c.startswith(("cos_", "sin_"))
                 and c not in HARMONIC_REMOVE]
    return obs + fut + harmonics


# ── ETL pipeline entry point ──────────────────────────────────────────────────

def build_training_dataset(
    obs_wl_richmond, pred_tide_richmond,
    obs_wl_fortpoint, pred_tide_fortpoint,
    obs_wl_pointreyes, pred_tide_pointreyes,
    pressure, sst, hfr, discharge, outflow, wind, upwelling,
    train_end: str = TRAIN_END,
    begin_iso: str = BEGIN_ISO,
    end_iso: str = END_ISO,
) -> tuple[pd.DataFrame, dict, float]:
    """Full ETL pipeline: merge → fit utide/PCA on train → engineer features.

    Returns:
        df          — full feature-engineered DataFrame (train + test rows)
        utide_coef  — utide coefficients fit on training period
        pca_angle   — PCA rotation angle fit on training period

    Note: utide_coef and pca_angle are fit on the full training period only.
    In CV, callers must refit these independently on each fold's training window.
    """
    logger.info("Merging data sources ...")
    df = merge_sources(
        obs_wl_richmond, pred_tide_richmond,
        obs_wl_fortpoint, pred_tide_fortpoint,
        obs_wl_pointreyes, pred_tide_pointreyes,
        pressure, sst, hfr, discharge, outflow, wind, upwelling,
        begin_iso=begin_iso, end_iso=end_iso,
    )

    # Fit utide and PCA on training period only
    train_mask = df.index <= train_end
    logger.info("Fitting utide on training period (%s rows) ...", train_mask.sum())
    utide_coef = fit_utide(df.loc[train_mask, "curr_u"], df.loc[train_mask, "curr_v"])

    logger.info("Fitting PCA rotation on training period ...")
    pca_angle = fit_pca_rotation(df.loc[train_mask, "curr_u"], df.loc[train_mask, "curr_v"])

    logger.info("Engineering features ...")
    df = engineer_features(df, utide_coef, pca_angle)

    logger.info("Dataset ready: %d rows × %d cols", *df.shape)
    return df, utide_coef, pca_angle
