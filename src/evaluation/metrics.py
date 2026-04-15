"""
Evaluation metrics for current forecast models.

All metrics operate on pandas Series with aligned DatetimeIndex.
Direction MAE uses circular statistics and is filtered to speeds > SLACK_SPEED_MS.
"""

import numpy as np
import pandas as pd

from src.config import SLACK_SPEED_MS, SPEED_CLIP_MS
from src.models.baseline import skill_score, tidal_baseline_mae


def reconstruct_speed(
    resid_along: pd.Series,
    resid_cross: pd.Series,
    tide_along: pd.Series,
    tide_cross: pd.Series,
) -> pd.Series:
    """Reconstruct full current speed from residual prediction + tidal component."""
    idx         = resid_along.index.intersection(tide_along.index)
    full_along  = resid_along.reindex(idx) + tide_along.reindex(idx)
    full_cross  = resid_cross.reindex(idx) + tide_cross.reindex(idx)
    speed       = np.sqrt(full_along ** 2 + full_cross ** 2)
    return speed.clip(0, SPEED_CLIP_MS)


def direction_mae(
    true_along: pd.Series,
    true_cross: pd.Series,
    pred_along: pd.Series,
    pred_cross: pd.Series,
    min_speed: float = SLACK_SPEED_MS,
) -> float:
    """Circular mean absolute error of current direction (degrees).

    Filtered to rows where true current speed > min_speed (NOAA slack definition).
    Never averages angles arithmetically — uses arctan2 on sin/cos differences.
    """
    idx = true_along.index.intersection(pred_along.index)
    ta, tc = true_along.reindex(idx), true_cross.reindex(idx)
    pa, pc = pred_along.reindex(idx), pred_cross.reindex(idx)

    true_spd = np.sqrt(ta ** 2 + tc ** 2)
    mask     = (true_spd > min_speed) & ta.notna() & tc.notna() & pa.notna() & pc.notna()

    true_dir = np.degrees(np.arctan2(tc[mask], ta[mask]))
    pred_dir = np.degrees(np.arctan2(pc[mask], pa[mask]))

    diff = true_dir - pred_dir
    # Wrap to [-180, 180]
    diff = (diff + 180) % 360 - 180
    return float(np.abs(diff).mean())


def evaluate_model(
    label: str,
    pred_along: pd.Series,
    pred_cross: pd.Series,
    true_along: pd.Series,
    true_cross: pd.Series,
    tide_along: pd.Series,
    tide_cross: pd.Series,
) -> dict:
    """Compute full evaluation suite for one model.

    Returns a dict of scalar metrics suitable for MLflow logging.
    """
    idx = (
        pred_along.index
        .intersection(pred_cross.index)
        .intersection(true_along.index)
        .intersection(true_cross.index)
    )
    pa = pred_along.reindex(idx)
    pc = pred_cross.reindex(idx)
    ta = true_along.reindex(idx)
    tc = true_cross.reindex(idx)

    valid = ta.notna() & tc.notna() & pa.notna() & pc.notna()
    pa, pc, ta, tc = pa[valid], pc[valid], ta[valid], tc[valid]

    mae_along = float(np.abs(pa - ta).mean())
    mae_cross = float(np.abs(pc - tc).mean())

    pred_spd = reconstruct_speed(pa, pc, tide_along, tide_cross)
    true_spd = reconstruct_speed(ta, tc, tide_along, tide_cross)
    speed_idx = pred_spd.index.intersection(true_spd.index)
    mae_speed = float(np.abs(pred_spd.reindex(speed_idx) - true_spd.reindex(speed_idx)).mean())

    mae_tide_along = tidal_baseline_mae(ta)
    mae_tide_cross = tidal_baseline_mae(tc)
    mae_tide_speed = tidal_baseline_mae(true_spd)

    dir_mae = direction_mae(ta, tc, pa, pc)

    return {
        f"{label}_mae_along":         mae_along,
        f"{label}_mae_cross":         mae_cross,
        f"{label}_mae_speed":         mae_speed,
        f"{label}_skill_along":       skill_score(mae_along, mae_tide_along),
        f"{label}_skill_cross":       skill_score(mae_cross, mae_tide_cross),
        f"{label}_skill_speed":       skill_score(mae_speed, mae_tide_speed),
        f"{label}_direction_mae_deg": dir_mae,
    }


def results_table(metrics_by_model: dict[str, dict]) -> pd.DataFrame:
    """Format a dict of {label: metrics_dict} into a readable comparison table."""
    rows = {}
    metric_keys = [
        "mae_along", "mae_cross", "mae_speed",
        "skill_along", "skill_cross", "skill_speed",
        "direction_mae_deg",
    ]
    for label, m in metrics_by_model.items():
        rows[label] = {k: m.get(f"{label}_{k}", float("nan")) for k in metric_keys}
    return pd.DataFrame(rows).T
