"""
ETL script for berkeley-circle-ml.

Fetches all data sources, builds the feature-engineered training dataset,
and saves inference artifacts to data/artifacts/. Run this once before
training; subsequent runs load from the parquet/CSV cache.

Usage:
    conda activate berkeley-circle-ml
    cd ~/Documents/github_repos/berkeley-circle-ml
    python src/run_etl.py

Outputs:
    data/berkeley_circle_train_<begin>_<end>.csv   — full feature dataset
    data/artifacts/utide_coef.pkl                 — tidal harmonic coefficients
    data/artifacts/pca_angle.pkl                  — along/cross rotation angle

Note: HFR fetch takes ~45 minutes on first run. Subsequent runs load from
data/cache parquet automatically.
"""

import gc
import logging
import pickle
from pathlib import Path

import pandas as pd

from src.config import (
    ARTIFACTS_DIR,
    BEGIN_ISO,
    COOPS_FORTPOINT,
    COOPS_POINTREYES,
    COOPS_RICHMOND,
    END_ISO,
    TRAIN_CSV_PATH,
)
from src.etl.features import build_training_dataset
from src.etl.fetch import (
    fetch_delta_outflow,
    fetch_discharge,
    fetch_hfr,
    fetch_pressure,
    fetch_tide_predictions,
    fetch_upwelling,
    fetch_water_level,
    fetch_wind,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
)
log = logging.getLogger(__name__)

# ── Resolve repo root ──────────────────────────────────────────────────────────
_repo_root = next(
    p for p in [Path(__file__).parent, *Path(__file__).parent.parents]
    if (p / 'pyproject.toml').exists()
)


def main() -> None:
    # ── Fetch ─────────────────────────────────────────────────────────────────
    log.info('Fetching Richmond water level + tide predictions ...')
    obs_wl_richmond    = fetch_water_level(COOPS_RICHMOND)
    pred_tide_richmond = fetch_tide_predictions(COOPS_RICHMOND)
    log.info('Richmond: done')

    log.info('Fetching Fort Point water level + tide predictions + pressure ...')
    obs_wl_fortpoint    = fetch_water_level(COOPS_FORTPOINT)
    pred_tide_fortpoint = fetch_tide_predictions(COOPS_FORTPOINT)
    pressure            = fetch_pressure()
    log.info('Fort Point: done')

    log.info('Fetching Point Reyes water level + tide predictions ...')
    obs_wl_pointreyes    = fetch_water_level(COOPS_POINTREYES)
    pred_tide_pointreyes = fetch_tide_predictions(COOPS_POINTREYES)
    log.info('Point Reyes: done')

    log.info('Fetching HFR currents (loads from cache if available) ...')
    hfr = fetch_hfr()
    log.info('HFR: done  %s', hfr.shape)

    log.info('Fetching wind + upwelling ...')
    wind      = fetch_wind()
    upwelling = fetch_upwelling()
    log.info('Wind + upwelling: done')

    log.info('Fetching discharge + delta outflow ...')
    discharge = fetch_discharge()
    outflow   = fetch_delta_outflow()
    log.info('Discharge + outflow: done')

    # SST not used (dropped after SHAP analysis)
    sst = pd.Series(dtype=float, name='sst_c')

    # ── Build dataset ─────────────────────────────────────────────────────────
    log.info('Building training dataset ...')
    df, utide_coef, pca_angle = build_training_dataset(
        obs_wl_richmond, pred_tide_richmond,
        obs_wl_fortpoint, pred_tide_fortpoint,
        obs_wl_pointreyes, pred_tide_pointreyes,
        pressure, sst, hfr, discharge, outflow, wind, upwelling,
    )

    # Free raw data — no longer needed
    del (obs_wl_richmond, pred_tide_richmond,
         obs_wl_fortpoint, pred_tide_fortpoint,
         obs_wl_pointreyes, pred_tide_pointreyes,
         pressure, sst, hfr, discharge, outflow, wind, upwelling)
    gc.collect()

    log.info('Dataset: %s rows × %s cols', *df.shape)

    # ── Save CSV ──────────────────────────────────────────────────────────────
    out_path = _repo_root / TRAIN_CSV_PATH.format(
        begin=BEGIN_ISO.replace('-', ''),
        end=END_ISO.replace('-', ''),
    )
    df.to_csv(out_path)
    log.info('Saved: %s', out_path)

    # ── Save artifacts ────────────────────────────────────────────────────────
    artifacts_dir = _repo_root / ARTIFACTS_DIR
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    for name, obj in [('utide_coef', utide_coef), ('pca_angle', pca_angle)]:
        path = artifacts_dir / f'{name}.pkl'
        with open(path, 'wb') as f:
            pickle.dump(obj, f)
        log.info('Saved: %s', path)

    log.info('ETL complete. Run `python src/train.py` to train models.')


if __name__ == '__main__':
    main()
