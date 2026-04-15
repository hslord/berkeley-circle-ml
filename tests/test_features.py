"""
Unit tests for ETL feature engineering functions.

Tests cover:
  - rotate_currents: round-trip and known angles
  - _fill_nulls: interpolation limits respected
  - skill_score: edge cases
  - direction_mae: circular correctness, slack filter
"""

import numpy as np
import pandas as pd
import pytest

from src.etl.features import _fill_nulls, rotate_currents
from src.evaluation.metrics import direction_mae
from src.models.baseline import skill_score

# ── rotate_currents ───────────────────────────────────────────────────────────

def test_rotate_currents_zero_angle():
    """At angle=0, along=u and cross=v (identity rotation)."""
    u = pd.Series([1.0, 0.0, -1.0])
    v = pd.Series([0.0, 1.0,  0.0])
    along, cross = rotate_currents(u, v, angle=0.0)
    np.testing.assert_allclose(along.values, u.values, atol=1e-10)
    np.testing.assert_allclose(cross.values, v.values, atol=1e-10)


def test_rotate_currents_90_degrees():
    """At angle=pi/2, a unit eastward vector maps to cross-channel."""
    u = pd.Series([1.0])
    v = pd.Series([0.0])
    along, cross = rotate_currents(u, v, angle=np.pi / 2)
    np.testing.assert_allclose(along.values, [0.0], atol=1e-10)
    np.testing.assert_allclose(cross.values, [-1.0], atol=1e-10)


def test_rotate_currents_roundtrip():
    """Rotating forward and backward recovers original u/v."""
    rng = np.random.default_rng(42)
    u = pd.Series(rng.normal(size=100))
    v = pd.Series(rng.normal(size=100))
    angle = 0.73  # arbitrary angle

    along, cross = rotate_currents(u, v, angle)
    # Inverse rotation: angle negated
    u_rec, v_rec = rotate_currents(along, -cross, -angle)
    # Note: rotate_currents(along, cross, -angle) inverts the rotation
    # along = cos(a)*u + sin(a)*v  →  u = cos(a)*along - sin(a)*cross
    # Verify via direct formula
    c, s = np.cos(angle), np.sin(angle)
    u_manual = c * along - s * cross
    v_manual = s * along + c * cross
    np.testing.assert_allclose(u_manual.values, u.values, atol=1e-10)
    np.testing.assert_allclose(v_manual.values, v.values, atol=1e-10)


def test_rotate_preserves_speed():
    """Rotation is orthogonal — speed is conserved."""
    rng   = np.random.default_rng(0)
    u     = pd.Series(rng.normal(size=50))
    v     = pd.Series(rng.normal(size=50))
    along, cross = rotate_currents(u, v, angle=1.23)
    speed_orig = np.sqrt(u ** 2 + v ** 2)
    speed_rot  = np.sqrt(along ** 2 + cross ** 2)
    np.testing.assert_allclose(speed_rot.values, speed_orig.values, atol=1e-10)


# ── _fill_nulls ───────────────────────────────────────────────────────────────

def _make_df_with_gaps():
    idx = pd.date_range("2020-01-01", periods=100, freq="h")
    df  = pd.DataFrame(index=idx)
    df["obs_wl"]        = 1.0
    df["pred_tide"]     = 1.0
    df["ntr"]           = 0.0
    df["dtide_dt"]      = 0.0
    df["ntr_fortpoint"] = 0.0
    df["ntr_pointreyes"]= 0.0
    df["ntr_gradient"]  = 0.0
    df["wind_u"]        = 0.5
    df["wind_v"]        = 0.5
    df["wind_spd_max_d"]= 0.5
    df["pressure_hpa"]  = 1013.0
    df["discharge_cms"] = 100.0
    df["outflow_cms"]   = 50.0
    df["sst_c"]         = 15.0
    df["upwelling_idx"] = 0.0
    df["curr_u"]        = np.nan   # targets — never filled
    df["curr_v"]        = np.nan
    return df


def test_fill_nulls_targets_not_filled():
    """curr_u and curr_v must remain NaN after _fill_nulls."""
    df = _make_df_with_gaps()
    filled = _fill_nulls(df)
    assert filled["curr_u"].isna().all()
    assert filled["curr_v"].isna().all()


def test_fill_nulls_tidal_limit():
    """Tidal columns: gaps > 6h should not be filled."""
    df = _make_df_with_gaps()
    df.loc[df.index[10:20], "ntr"] = np.nan   # 10-hour gap
    filled = _fill_nulls(df)
    # First 6 hours after gap start should be filled, remainder NaN
    assert filled["ntr"].iloc[10:16].notna().all()
    assert filled["ntr"].iloc[16:20].isna().any()


# ── skill_score ───────────────────────────────────────────────────────────────

def test_skill_score_perfect():
    assert skill_score(0.0, 1.0) == pytest.approx(1.0)


def test_skill_score_baseline_equal():
    assert skill_score(1.0, 1.0) == pytest.approx(0.0)


def test_skill_score_worse_than_baseline():
    assert skill_score(2.0, 1.0) == pytest.approx(-1.0)


def test_skill_score_zero_baseline():
    assert np.isnan(skill_score(0.5, 0.0))


# ── direction_mae ──────────────────────────────────────────────────────────────

def test_direction_mae_perfect():
    """Perfect predictions → direction MAE = 0."""
    idx   = pd.RangeIndex(10)
    along = pd.Series([1.0] * 10, index=idx)
    cross = pd.Series([0.0] * 10, index=idx)
    assert direction_mae(along, cross, along, cross) == pytest.approx(0.0)


def test_direction_mae_opposite():
    """180° off predictions → direction MAE = 180."""
    idx        = pd.RangeIndex(10)
    true_along = pd.Series([1.0] * 10, index=idx)
    true_cross = pd.Series([0.0] * 10, index=idx)
    pred_along = pd.Series([-1.0] * 10, index=idx)
    pred_cross = pd.Series([0.0] * 10, index=idx)
    assert direction_mae(true_along, true_cross, pred_along, pred_cross) == pytest.approx(180.0)


def test_direction_mae_slack_filter():
    """Rows below slack speed threshold should be excluded."""
    idx        = pd.RangeIndex(4)
    # First two rows: speed = 0.01 (below SLACK_SPEED_MS=0.05) → excluded
    # Last two rows: speed = 1.0, 180° error
    true_along = pd.Series([0.01, 0.01, 1.0, 1.0], index=idx)
    true_cross = pd.Series([0.0,  0.0,  0.0, 0.0], index=idx)
    pred_along = pd.Series([0.01, 0.01, -1.0, -1.0], index=idx)
    pred_cross = pd.Series([0.0,  0.0,   0.0,  0.0], index=idx)
    mae = direction_mae(true_along, true_cross, pred_along, pred_cross)
    assert mae == pytest.approx(180.0)
