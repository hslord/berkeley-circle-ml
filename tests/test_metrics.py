"""
Unit tests for evaluation metrics.

Tests cover:
  - reconstruct_speed: correct formula, clipping, index alignment
  - evaluate_model: metric keys, skill bounds, direction MAE
"""

import numpy as np
import pandas as pd
import pytest

from src.config import SPEED_CLIP_MS
from src.evaluation.metrics import evaluate_model, reconstruct_speed

# ── Helpers ───────────────────────────────────────────────────────────────────

def _series(values, start="2025-01-01"):
    idx = pd.date_range(start, periods=len(values), freq="h")
    return pd.Series(values, index=idx, dtype=float)


# ── reconstruct_speed ─────────────────────────────────────────────────────────

def test_reconstruct_speed_pure_tidal():
    """With zero residual, speed equals tidal speed."""
    tide_along = _series([0.5, 0.3, -0.2])
    tide_cross = _series([0.1, 0.0,  0.1])
    resid = _series([0.0, 0.0, 0.0])

    speed = reconstruct_speed(resid, resid, tide_along, tide_cross)
    expected = np.sqrt(tide_along.values ** 2 + tide_cross.values ** 2)
    np.testing.assert_allclose(speed.values, expected, atol=1e-10)


def test_reconstruct_speed_clipped():
    """Speed above SPEED_CLIP_MS is clipped."""
    tide_along = _series([2.0])
    tide_cross = _series([0.0])
    resid      = _series([0.0])

    speed = reconstruct_speed(resid, resid, tide_along, tide_cross)
    assert speed.iloc[0] == pytest.approx(SPEED_CLIP_MS)


def test_reconstruct_speed_index_intersection():
    """Only overlapping timestamps are used."""
    tide_along = _series([0.3, 0.4, 0.5])
    tide_cross = _series([0.0, 0.0, 0.0])
    # Residuals only cover first two timestamps
    resid_short = pd.Series(
        [0.0, 0.0],
        index=pd.date_range("2025-01-01", periods=2, freq="h"),
    )

    speed = reconstruct_speed(resid_short, resid_short, tide_along, tide_cross)
    assert len(speed) == 2


# ── evaluate_model ────────────────────────────────────────────────────────────

def _perfect_eval():
    """evaluate_model with perfect predictions."""
    idx        = pd.date_range("2025-01-01", periods=100, freq="h")
    true_along = pd.Series(np.random.default_rng(0).normal(0, 0.1, 100), index=idx)
    true_cross = pd.Series(np.random.default_rng(1).normal(0, 0.05, 100), index=idx)
    tide_along = pd.Series(np.sin(np.linspace(0, 4 * np.pi, 100)) * 0.4, index=idx)
    tide_cross = pd.Series(np.zeros(100), index=idx)
    return true_along, true_cross, tide_along, tide_cross


def test_evaluate_model_keys():
    """evaluate_model returns all expected metric keys."""
    ta, tc, tida, tidc = _perfect_eval()
    metrics = evaluate_model('xgb', ta, tc, ta, tc, tida, tidc)

    expected_keys = {
        'xgb_mae_along', 'xgb_mae_cross', 'xgb_mae_speed',
        'xgb_skill_along', 'xgb_skill_cross', 'xgb_skill_speed',
        'xgb_direction_mae_deg',
    }
    assert expected_keys == set(metrics.keys())


def test_evaluate_model_perfect_skill():
    """Perfect predictions → skill = 1 for along and cross."""
    ta, tc, tida, tidc = _perfect_eval()
    metrics = evaluate_model('m', ta, tc, ta, tc, tida, tidc)

    assert metrics['m_skill_along'] == pytest.approx(1.0, abs=1e-6)
    assert metrics['m_skill_cross'] == pytest.approx(1.0, abs=1e-6)


def test_evaluate_model_skill_bounded():
    """Skill scores are ≤ 1 for any predictions."""
    ta, tc, tida, tidc = _perfect_eval()
    rng  = np.random.default_rng(42)
    idx  = ta.index
    pred_along = pd.Series(rng.normal(0, 0.3, len(idx)), index=idx)
    pred_cross = pd.Series(rng.normal(0, 0.1, len(idx)), index=idx)

    metrics = evaluate_model('m', pred_along, pred_cross, ta, tc, tida, tidc)
    assert metrics['m_skill_along'] <= 1.0
    assert metrics['m_skill_cross'] <= 1.0
    assert metrics['m_skill_speed'] <= 1.0


def test_evaluate_model_mae_nonnegative():
    """MAE values are always ≥ 0."""
    ta, tc, tida, tidc = _perfect_eval()
    rng  = np.random.default_rng(7)
    idx  = ta.index
    pred_along = pd.Series(rng.normal(0, 0.2, len(idx)), index=idx)
    pred_cross = pd.Series(rng.normal(0, 0.1, len(idx)), index=idx)

    metrics = evaluate_model('m', pred_along, pred_cross, ta, tc, tida, tidc)
    assert metrics['m_mae_along'] >= 0
    assert metrics['m_mae_cross'] >= 0
    assert metrics['m_mae_speed'] >= 0


def test_evaluate_model_direction_mae_range():
    """Direction MAE is in [0, 180] degrees."""
    ta, tc, tida, tidc = _perfect_eval()
    rng        = np.random.default_rng(3)
    idx        = ta.index
    pred_along = pd.Series(rng.normal(0, 0.3, len(idx)), index=idx)
    pred_cross = pd.Series(rng.normal(0, 0.1, len(idx)), index=idx)

    metrics = evaluate_model('m', pred_along, pred_cross, ta, tc, tida, tidc)
    assert 0 <= metrics['m_direction_mae_deg'] <= 180
