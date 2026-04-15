"""
Training script for richmond-current-ml.

Runs CV + XGBoost + LSTM + MLflow logging.

Usage:
    conda activate richmond-current-ml
    cd ~/Documents/github_repos/richmond-current-ml
    python src/train.py

MLflow results appear in mlruns/ and are visible in the UI at http://localhost:5001
"""

import logging
import os
import pickle
import tempfile
from pathlib import Path

import mlflow
import mlflow.pytorch
import mlflow.xgboost
import pandas as pd
import shap as shap_lib

from src.config import (
    ARTIFACTS_DIR,
    BEGIN_ISO,
    END_ISO,
    FORECAST_HOURS,
    LSTM_BATCH,
    LSTM_DROPOUT,
    LSTM_HIDDEN,
    LSTM_LAYERS,
    LSTM_LOOKBACK,
    LSTM_VAL_YEAR,
    MLFLOW_EXPERIMENT,
    MLFLOW_RUN_NAME,
    PLOT_WINDOW_END,
    PLOT_WINDOW_START,
    TEST_START,
    TRAIN_CSV_PATH,
    TRAIN_END,
    XGB_PARAMS,
)
from src.etl.features import get_feature_cols
from src.evaluation.metrics import evaluate_model, reconstruct_speed, results_table
from src.evaluation.plots import (
    plot_cv_skill,
    plot_loss_curve,
    plot_scatter,
    plot_shap_beeswarm,
    plot_shap_summary,
    plot_time_series,
)
from src.models.baseline import tidal_baseline_mae
from src.models.lstm_model import predict_lstm, train_lstm
from src.models.xgboost_model import run_cv, train_quantile_xgboost, train_xgboost

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
)
log = logging.getLogger(__name__)

# ── Resolve repo root and MLflow path ─────────────────────────────────────────
_repo_root = next(
    p for p in [Path(__file__).parent, *Path(__file__).parent.parents]
    if (p / 'pyproject.toml').exists()
)
MLRUNS_PATH = str(_repo_root / 'mlruns')
mlflow.set_tracking_uri(MLRUNS_PATH)
mlflow.set_experiment(MLFLOW_EXPERIMENT)


def main() -> None:
    # ── Load dataset ──────────────────────────────────────────────────────────
    csv_path = _repo_root / TRAIN_CSV_PATH.format(
        begin=BEGIN_ISO.replace('-', ''),
        end=END_ISO.replace('-', ''),
    )
    log.info('Loading dataset from %s', csv_path)
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    feature_cols = get_feature_cols(df)
    log.info('Dataset: %s rows × %s cols, %s features', len(df), df.shape[1], len(feature_cols))

    # Load inference artifacts (utide_coef, pca_angle) from most recent MLflow run
    artifact_hits = sorted(
        Path(MLRUNS_PATH).glob('*/*/artifacts/inference_artifacts/utide_coef.pkl')
    )
    if artifact_hits:
        _adir = artifact_hits[-1].parent
        log.info('Loading inference artifacts from %s', _adir)
        with open(_adir / 'utide_coef.pkl', 'rb') as f:
            utide_coef = pickle.load(f)
        with open(_adir / 'pca_angle.pkl', 'rb') as f:
            pca_angle = pickle.load(f)
    else:
        log.warning('No pre-computed artifacts found — utide_coef and pca_angle will be None')
        utide_coef = None
        pca_angle  = None

    train = df.loc[:TRAIN_END]
    test  = df.loc[TEST_START:]
    log.info('Train: %s rows  Test: %s rows', len(train), len(test))

    # ── MLflow run ────────────────────────────────────────────────────────────
    if mlflow.active_run():
        mlflow.end_run()

    with mlflow.start_run(run_name=MLFLOW_RUN_NAME) as run:
        log.info('MLflow run: %s', run.info.run_id)

        mlflow.log_params({
            'forecast_hours': FORECAST_HOURS,
            'train_end':      TRAIN_END,
            'test_start':     TEST_START,
            'n_features':     len(feature_cols),
            'train_rows':     len(train),
            'test_rows':      len(test),
        })
        mlflow.log_params({
            f'xgb_{k}': v for k, v in XGB_PARAMS.items()
            if k not in ('eval_metric', 'early_stopping_rounds', 'n_jobs', 'random_state')
        })
        mlflow.log_params({
            'lstm_lookback': LSTM_LOOKBACK,
            'lstm_hidden':   LSTM_HIDDEN,
            'lstm_layers':   LSTM_LAYERS,
            'lstm_dropout':  LSTM_DROPOUT,
            'lstm_batch':    LSTM_BATCH,
            'lstm_val_year': LSTM_VAL_YEAR,
        })

        # ── Cross-validation ──────────────────────────────────────────────────
        log.info('Running expanding-window CV ...')
        cv_df = run_cv(df)
        print('\n── CV Results ──')
        print(cv_df.to_string(float_format='{:.3f}'.format))
        mlflow.log_figure(plot_cv_skill(cv_df), 'cv_skill.png')

        # ── Final XGBoost ─────────────────────────────────────────────────────
        log.info('Training final XGBoost ...')
        val_tr = train[train.index.year < LSTM_VAL_YEAR]
        val_va = train[train.index.year >= LSTM_VAL_YEAR]
        xgb_models   = train_xgboost(val_tr, val_va, feature_cols)
        xgb_q_models = train_quantile_xgboost(val_tr, val_va, feature_cols)

        mlflow.xgboost.log_model(xgb_models['resid_along_fwd'], 'xgb_along')
        mlflow.xgboost.log_model(xgb_models['resid_cross_fwd'], 'xgb_cross')

        # ── LSTM ──────────────────────────────────────────────────────────────
        log.info('Training LSTM ...')
        lstm_model, feat_scaler, target_scaler, history = train_lstm(train, feature_cols)
        mlflow.pytorch.log_model(lstm_model, 'lstm')
        mlflow.log_figure(
            plot_loss_curve(history['train_losses'], history['val_losses'], history['best_epoch']),
            'lstm_loss_curve.png',
        )

        # ── Save inference artifacts ───────────────────────────────────────────
        with tempfile.TemporaryDirectory() as tmp:
            for name, obj in [
                ('feat_scaler',    feat_scaler),
                ('target_scaler',  target_scaler),
                ('pca_angle',      pca_angle),
                ('utide_coef',     utide_coef),
            ]:
                path = os.path.join(tmp, f'{name}.pkl')
                with open(path, 'wb') as f:
                    pickle.dump(obj, f)
                mlflow.log_artifact(path, 'inference_artifacts')

        # Also write to data/artifacts/ for stable notebook loading
        artifacts_dir = _repo_root / ARTIFACTS_DIR
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        for name, obj in [('utide_coef', utide_coef), ('pca_angle', pca_angle),
                           ('feat_scaler', feat_scaler), ('target_scaler', target_scaler)]:
            with open(artifacts_dir / f'{name}.pkl', 'wb') as f:
                pickle.dump(obj, f)
        log.info('Artifacts saved to %s', artifacts_dir)

        # ── Test set predictions ──────────────────────────────────────────────
        fh   = FORECAST_HOURS
        X_te = test[feature_cols]

        xgb_pred_along = pd.Series(
            xgb_models['resid_along_fwd'].predict(X_te), index=test.index, name='xgb_resid_along'
        )
        xgb_pred_cross = pd.Series(
            xgb_models['resid_cross_fwd'].predict(X_te), index=test.index, name='xgb_resid_cross'
        )
        lstm_pred_along, lstm_pred_cross = predict_lstm(
            lstm_model, train, test, feat_scaler, target_scaler, feature_cols
        )

        tide_along = test[f'tide_curr_along_{fh}h']
        tide_cross = test[f'tide_curr_cross_{fh}h']
        true_along = test['resid_along_fwd'].dropna()
        true_cross = test['resid_cross_fwd'].dropna()

        # ── Baseline metrics ──────────────────────────────────────────────────
        true_spd = reconstruct_speed(true_along, true_cross, tide_along, tide_cross)
        mlflow.log_metrics({
            'baseline_mae_along': tidal_baseline_mae(true_along),
            'baseline_mae_cross': tidal_baseline_mae(true_cross),
            'baseline_mae_speed': tidal_baseline_mae(true_spd),
        })

        # ── Model metrics ─────────────────────────────────────────────────────
        xgb_metrics  = evaluate_model('xgboost', xgb_pred_along, xgb_pred_cross,
                                      true_along, true_cross, tide_along, tide_cross)
        lstm_metrics = evaluate_model('lstm', lstm_pred_along, lstm_pred_cross,
                                      true_along, true_cross, tide_along, tide_cross)

        # Ensemble: equal-weight average of XGBoost and LSTM
        ens_pred_along = (xgb_pred_along + lstm_pred_along) / 2
        ens_pred_cross = (xgb_pred_cross + lstm_pred_cross) / 2
        ens_metrics    = evaluate_model('ensemble', ens_pred_along, ens_pred_cross,
                                        true_along, true_cross, tide_along, tide_cross)

        mlflow.log_metrics({**xgb_metrics, **lstm_metrics, **ens_metrics})

        print('\n── Test Set Results ──')
        print(results_table({'xgboost': xgb_metrics, 'lstm': lstm_metrics, 'ensemble': ens_metrics})
              .to_string(float_format='{:.3f}'.format))

        # ── Plots ─────────────────────────────────────────────────────────────
        q10_along = pd.Series(
            xgb_q_models[('resid_along_fwd', 0.1)].predict(X_te), index=test.index
        )
        q90_along = pd.Series(
            xgb_q_models[('resid_along_fwd', 0.9)].predict(X_te), index=test.index
        )

        xgb_spd  = reconstruct_speed(xgb_pred_along, xgb_pred_cross, tide_along, tide_cross)
        lstm_spd = reconstruct_speed(lstm_pred_along, lstm_pred_cross, tide_along, tide_cross)

        mlflow.log_figure(
            plot_time_series(test, xgb_pred_along, tide_along, true_along, label='XGBoost',
                             window_start=PLOT_WINDOW_START, window_end=PLOT_WINDOW_END,
                             q10=q10_along, q90=q90_along),
            'time_series_xgb_feb2025.png',
        )
        ens_spd = reconstruct_speed(ens_pred_along, ens_pred_cross, tide_along, tide_cross)
        mlflow.log_figure(plot_scatter(true_spd, xgb_spd,  'XGBoost'),             'scatter_xgb.png')
        mlflow.log_figure(plot_scatter(true_spd, lstm_spd, 'LSTM', 'steelblue'),   'scatter_lstm.png')
        mlflow.log_figure(plot_scatter(true_spd, ens_spd,  'Ensemble', 'seagreen'), 'scatter_ensemble.png')

        # ── SHAP ──────────────────────────────────────────────────────────────
        log.info('Computing SHAP ...')
        for target, label in [('resid_along_fwd', 'along'), ('resid_cross_fwd', 'cross')]:
            explainer   = shap_lib.TreeExplainer(xgb_models[target])
            shap_values = explainer.shap_values(X_te)
            mlflow.log_figure(plot_shap_summary(shap_values, feature_cols, label),
                              f'shap_bar_{label}.png')
            mlflow.log_figure(plot_shap_beeswarm(shap_values, X_te, label),
                              f'shap_beeswarm_{label}.png')

        mlflow.set_tags({
            'status':     'completed',
            'best_model': 'xgboost'
                          if xgb_metrics['xgboost_skill_speed'] >= lstm_metrics['lstm_skill_speed']
                          else 'lstm',
        })

    log.info('Run ID: %s', run.info.run_id)
    log.info('View results: http://localhost:5001')


if __name__ == '__main__':
    main()
