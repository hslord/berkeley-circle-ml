"""
Inference script for berkeley-circle-ml.

Produces a 24-hour-ahead current forecast for a given issue time using the
trained XGBoost models and pre-built feature dataset. Operates on historical
data — see README for inference limitations.

Usage:
    python src/predict.py                        # most recent timestamp in dataset
    python src/predict.py "2025-04-01 12:00"    # specific issue time

Requires:
    python src/run_etl.py   # to build the feature dataset
    python src/train.py     # to train and log models
"""

import argparse
import sys
from pathlib import Path

import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd

from src.config import (
    BEGIN_ISO,
    END_ISO,
    FORECAST_HOURS,
    MLFLOW_EXPERIMENT,
    TRAIN_CSV_PATH,
)
from src.etl.features import get_feature_cols

_repo_root = next(
    p for p in [Path(__file__).parent, *Path(__file__).parent.parents]
    if (p / 'pyproject.toml').exists()
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description='24h current forecast for Berkeley Circle'
    )
    parser.add_argument(
        'time',
        nargs='?',
        help='Issue time, e.g. "2025-04-01 12:00". Defaults to most recent in dataset.',
    )
    args = parser.parse_args()

    # ── Load dataset ──────────────────────────────────────────────────────────
    csv_path = _repo_root / TRAIN_CSV_PATH.format(
        begin=BEGIN_ISO.replace('-', ''),
        end=END_ISO.replace('-', ''),
    )
    if not csv_path.exists():
        sys.exit(f'Dataset not found: {csv_path}\nRun `python src/run_etl.py` first.')

    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    feature_cols = get_feature_cols(df)

    # ── Resolve issue time ────────────────────────────────────────────────────
    if args.time:
        try:
            t = pd.Timestamp(args.time)
        except Exception:
            sys.exit(f'Could not parse time: {args.time!r}')
        if t not in df.index:
            sys.exit(
                f'Timestamp {t} not in dataset.\n'
                f'Dataset covers {df.index[0]} → {df.index[-1]}.'
            )
        if pd.isna(df.loc[t, f'tide_curr_along_{FORECAST_HOURS}h']):
            sys.exit(
                f'Timestamp {t} is too close to the end of the dataset — '
                f'tidal prediction at T+{FORECAST_HOURS}h is unavailable.\n'
                f'Choose a time at least {FORECAST_HOURS}h before {df.index[-1]}.'
            )
    else:
        fh_cols = [f'tide_curr_along_{FORECAST_HOURS}h', f'tide_curr_cross_{FORECAST_HOURS}h']
        t = df[fh_cols].dropna().index[-1]

    # ── Load models ───────────────────────────────────────────────────────────
    mlflow.set_tracking_uri(str(_repo_root / 'mlruns'))
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    runs = mlflow.search_runs(
        experiment_names=[MLFLOW_EXPERIMENT],
        filter_string="tags.status = 'completed'",
        order_by=['start_time DESC'],
        max_results=1,
    )
    if runs.empty:
        sys.exit('No completed MLflow runs found. Run `python src/train.py` first.')

    run_uri = f"runs:/{runs.iloc[0]['run_id']}"
    xgb_along = mlflow.xgboost.load_model(f'{run_uri}/xgb_along')
    xgb_cross = mlflow.xgboost.load_model(f'{run_uri}/xgb_cross')

    # ── Predict ───────────────────────────────────────────────────────────────
    fh   = FORECAST_HOURS
    X    = df.loc[[t], feature_cols]

    resid_along = float(xgb_along.predict(X)[0])
    resid_cross = float(xgb_cross.predict(X)[0])
    tide_along  = float(df.loc[t, f'tide_curr_along_{fh}h'])
    tide_cross  = float(df.loc[t, f'tide_curr_cross_{fh}h'])
    speed       = float(np.clip(
        np.sqrt((resid_along + tide_along) ** 2 + (resid_cross + tide_cross) ** 2),
        0, 1.5,
    ))
    direction   = float(np.degrees(np.arctan2(
        resid_cross + tide_cross,
        resid_along + tide_along,
    )) % 360)

    # ── Output ────────────────────────────────────────────────────────────────
    valid_time = t + pd.Timedelta(hours=fh)
    print('\n24h Forecast — Berkeley Circle')
    print(f'  Issue time : {t}')
    print(f'  Valid time : {valid_time}')
    print(f'  Speed      : {speed:.2f} m/s')
    print(f'  Direction  : {direction:.0f}°')
    print(f'  (along residual: {resid_along:+.3f} m/s, cross residual: {resid_cross:+.3f} m/s)')


if __name__ == '__main__':
    main()
