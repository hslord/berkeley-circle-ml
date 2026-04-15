"""
Baseline models for comparison against ML approaches.

Tidal-only baseline: predict zero residual (i.e., the full current is
just the tidal reconstruction). This is the primary benchmark — a skill
score of 0 means the ML model offers no improvement over harmonics alone.

Persistence baseline: predict the current residual at T as the residual
at T+24. Useful to confirm the ML model learns more than simple persistence.
"""

import numpy as np
import pandas as pd


def tidal_baseline_mae(true_resid: pd.Series) -> float:
    """MAE of tidal-only prediction for a residual target.

    Since the tidal-only model predicts residual = 0, MAE = mean(|true_resid|).
    """
    return float(np.abs(true_resid.dropna()).mean())


def persistence_baseline(resid_at_t: pd.Series, forecast_hours: int = 24) -> pd.Series:
    """Predict the residual at T+forecast_hours as the residual at T."""
    return resid_at_t.shift(-forecast_hours)


def skill_score(mae_model: float, mae_baseline: float) -> float:
    """Skill score: 1 - (MAE_model / MAE_baseline).

    0 = no improvement over baseline. Positive = better than baseline.
    """
    if mae_baseline == 0:
        return float("nan")
    return 1.0 - mae_model / mae_baseline
