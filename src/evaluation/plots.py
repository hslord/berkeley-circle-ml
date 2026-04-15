"""
Visualization functions for model evaluation.

All functions return matplotlib Figure objects so callers can log them
to MLflow via mlflow.log_figure() or display inline in notebooks.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


def plot_time_series(
    test_df: pd.DataFrame,
    pred_along: pd.Series,
    tide_along: pd.Series,
    true_resid_along: pd.Series,
    label: str,
    window_start: str = None,
    window_end: str = None,
    q10: pd.Series = None,
    q90: pd.Series = None,
) -> plt.Figure:
    """Time series of predicted vs observed along-channel current with optional uncertainty band."""
    true_full = true_resid_along + tide_along.reindex(true_resid_along.index)
    pred_full = pred_along + tide_along.reindex(pred_along.index)

    if window_start and window_end:
        true_full = true_full.loc[window_start:window_end]
        pred_full = pred_full.loc[window_start:window_end]
        if q10 is not None:
            q10 = q10.loc[window_start:window_end]
            q90 = q90.loc[window_start:window_end]

    fig, ax = plt.subplots(figsize=(14, 4))

    if q10 is not None and q90 is not None:
        q10_full = q10 + tide_along.reindex(q10.index)
        q90_full = q90 + tide_along.reindex(q90.index)
        ax.fill_between(q10_full.index, q10_full, q90_full,
                        alpha=0.18, color="tomato", label="80% interval (q10–q90)")

    ax.plot(true_full.index, true_full, "k", lw=1.2, label="Observed", zorder=3)
    ax.plot(pred_full.index, pred_full, color="tomato", lw=1.0, label=label, zorder=2)

    ax.set_ylabel("Along-channel current (m/s)")
    ax.set_title(f"{label} — 24h forecast vs observed")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_scatter(
    true_spd: pd.Series,
    pred_spd: pd.Series,
    label: str,
    color: str = "tomato",
) -> plt.Figure:
    """Predicted vs observed speed scatter plot."""
    idx  = true_spd.index.intersection(pred_spd.index)
    t    = true_spd.reindex(idx).dropna()
    p    = pred_spd.reindex(t.index)
    lim  = max(t.max(), p.max()) * 1.05

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(t, p, s=2, alpha=0.2, color=color, edgecolors="none")
    ax.plot([0, lim], [0, lim], "k-", lw=0.8, label="1:1")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Observed speed (m/s)")
    ax.set_ylabel("Predicted speed (m/s)")
    ax.set_title(label)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_cv_skill(cv_df: pd.DataFrame) -> plt.Figure:
    """Bar chart of CV skill score by fold."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, col, title in [
        (axes[0], "skill_along", "Along-channel"),
        (axes[1], "skill_cross", "Cross-channel"),
    ]:
        colors = ["tomato" if s > 0 else "steelblue" for s in cv_df[col]]
        ax.bar(cv_df.index, cv_df[col], color=colors, edgecolor="none")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_title(f"CV skill score — {title}")
        ax.set_xlabel("Validation year")
        ax.set_ylabel("Skill score (vs tidal only)")
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_shap_summary(
    shap_values: np.ndarray,
    feature_cols: list[str],
    target_label: str,
    top_n: int = 20,
) -> plt.Figure:
    """Mean |SHAP| bar chart for top N features."""
    mean_shap = pd.Series(np.abs(shap_values).mean(axis=0), index=feature_cols)
    mean_shap = mean_shap.sort_values(ascending=True).tail(top_n)

    fig, ax = plt.subplots(figsize=(8, 6))
    mean_shap.plot.barh(ax=ax, color="tomato", edgecolor="none")
    ax.set_title(f"Mean |SHAP| — {target_label}")
    ax.set_xlabel("Mean |SHAP value|")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_shap_beeswarm(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    target_label: str,
) -> plt.Figure:
    """SHAP beeswarm plot (summary plot with feature values)."""
    fig, ax = plt.subplots(figsize=(8, 8))
    shap.summary_plot(shap_values, X, show=False, plot_size=None)
    ax = plt.gca()
    ax.set_title(f"SHAP summary — {target_label}")
    fig = plt.gcf()
    fig.tight_layout()
    return fig


def plot_loss_curve(
    train_losses: list[float],
    val_losses: list[float],
    best_epoch: int,
) -> plt.Figure:
    """LSTM training vs validation loss curve."""
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(train_losses, label="Train")
    ax.plot(val_losses,   label="Val")
    ax.axvline(best_epoch - 1, color="red", ls="--", lw=0.8,
               label=f"Best epoch ({best_epoch})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss")
    ax.set_title("LSTM training vs validation loss")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig
