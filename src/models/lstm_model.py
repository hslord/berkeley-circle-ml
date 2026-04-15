"""
LSTM model (PyTorch) for along-channel and cross-channel residual prediction.

Architecture: 1-layer LSTM, hidden=64, dropout=0, lookback=48h.
A 168h lookback is an open experiment — see CLAUDE.md.

Scaler leakage fix: StandardScaler is fit only on the pre-val portion of
training data (years < LSTM_VAL_YEAR), not the full training period.
"""

import logging

import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from src.config import (
    LSTM_BATCH,
    LSTM_DROPOUT,
    LSTM_EPOCHS,
    LSTM_HIDDEN,
    LSTM_LAYERS,
    LSTM_LOOKBACK,
    LSTM_PATIENCE,
    LSTM_VAL_YEAR,
)

logger = logging.getLogger(__name__)

TARGETS = ["resid_along_fwd", "resid_cross_fwd"]


# ── Dataset ───────────────────────────────────────────────────────────────────

class CurrentDataset(Dataset):
    """Sliding window dataset. Each sample: (lookback × features) → (2,)."""

    def __init__(self, X: np.ndarray, y: np.ndarray, lookback: int):
        self.X        = X
        self.y        = y
        self.lookback = lookback

    def __len__(self) -> int:
        return len(self.X) - self.lookback

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.tensor(self.X[i : i + self.lookback], dtype=torch.float32)
        y = torch.tensor(self.y[i + self.lookback],     dtype=torch.float32)
        return x, y


# ── Model ─────────────────────────────────────────────────────────────────────

class CurrentLSTM(nn.Module):
    def __init__(self, n_features: int,
                 hidden: int = LSTM_HIDDEN,
                 n_layers: int = LSTM_LAYERS,
                 dropout: float = LSTM_DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(
            n_features, hidden, n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden, 2)   # along + cross

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


# ── Training ──────────────────────────────────────────────────────────────────

def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_lstm(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    lookback: int = LSTM_LOOKBACK,
    val_year: int = LSTM_VAL_YEAR,
) -> tuple[CurrentLSTM, StandardScaler, StandardScaler, dict]:
    """Train LSTM with early stopping.

    Scaler fit on pre-val training data only (years < val_year) to avoid
    leakage of val-period statistics into feature scaling.

    Returns:
        model          — trained CurrentLSTM (best checkpoint loaded)
        feat_scaler    — StandardScaler fit on pre-val training features
        target_scaler  — StandardScaler fit on pre-val training targets
        history        — dict with train_losses, val_losses, best_epoch
    """
    device = _get_device()
    logger.info("LSTM device: %s", device)

    # Fall back to last year in training data if val_year exceeds the range
    max_train_year = train_df.index.year.max()
    if val_year > max_train_year:
        val_year = max_train_year
        logger.warning("LSTM_VAL_YEAR exceeds training data; using %d", val_year)

    pre_val  = train_df[train_df.index.year < val_year]
    post_val = train_df[train_df.index.year >= val_year]

    # Scalers fit on pre-val only
    feat_scaler   = StandardScaler()
    target_scaler = StandardScaler()

    X_pre = feat_scaler.fit_transform(pre_val[feature_cols].fillna(0))
    y_pre = target_scaler.fit_transform(pre_val[TARGETS].fillna(0))

    X_post = feat_scaler.transform(post_val[feature_cols].fillna(0))
    y_post = target_scaler.transform(post_val[TARGETS].fillna(0))

    # Prepend lookback context to val so sequences have full history
    tr_ds = CurrentDataset(X_pre, y_pre, lookback)
    va_ds = CurrentDataset(
        np.vstack([X_pre[-lookback:], X_post]),
        np.vstack([y_pre[-lookback:], y_post]),
        lookback,
    )
    tr_dl = DataLoader(tr_ds, batch_size=LSTM_BATCH, shuffle=True)
    va_dl = DataLoader(va_ds, batch_size=LSTM_BATCH, shuffle=False)

    model     = CurrentLSTM(n_features=len(feature_cols)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    criterion = nn.MSELoss()

    best_val_loss    = float("inf")
    best_model_state = None
    patience_count   = 0
    train_losses, val_losses = [], []

    for epoch in range(1, LSTM_EPOCHS + 1):
        model.train()
        epoch_loss = sum(
            _train_step(model, X_b.to(device), y_b.to(device), optimizer, criterion)
            for X_b, y_b in tr_dl
        ) / len(tr_dl)

        model.eval()
        with torch.no_grad():
            val_loss = sum(
                criterion(model(X_b.to(device)), y_b.to(device)).item()
                for X_b, y_b in va_dl
            ) / len(va_dl)

        scheduler.step(val_loss)
        train_losses.append(epoch_loss)
        val_losses.append(val_loss)
        mlflow.log_metrics({"lstm_train_loss": epoch_loss, "lstm_val_loss": val_loss}, step=epoch)

        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_count   = 0
        else:
            patience_count += 1

        if epoch % 10 == 0:
            logger.info("Epoch %3d/%d  train=%.6f  val=%.6f  patience=%d/%d",
                        epoch, LSTM_EPOCHS, epoch_loss, val_loss, patience_count, LSTM_PATIENCE)

        if patience_count >= LSTM_PATIENCE:
            logger.info("Early stopping at epoch %d (best val=%.6f)", epoch, best_val_loss)
            break

    best_epoch = len(train_losses) - patience_count
    mlflow.log_metrics({"lstm_best_epoch": best_epoch, "lstm_best_val_loss": best_val_loss})

    model.load_state_dict(best_model_state)
    return model, feat_scaler, target_scaler, {
        "train_losses": train_losses,
        "val_losses":   val_losses,
        "best_epoch":   best_epoch,
        "best_val_loss": best_val_loss,
    }


def _train_step(model, X_b, y_b, optimizer, criterion) -> float:
    optimizer.zero_grad()
    loss = criterion(model(X_b), y_b)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return loss.item()


# ── Inference ─────────────────────────────────────────────────────────────────

def predict_lstm(
    model: CurrentLSTM,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feat_scaler: StandardScaler,
    target_scaler: StandardScaler,
    feature_cols: list[str],
    lookback: int = LSTM_LOOKBACK,
) -> tuple[pd.Series, pd.Series]:
    """Generate LSTM predictions on the test set.

    Prepends train tail to test so the first test window has full context.
    Returns (pred_along, pred_cross) as unscaled Series on the test index.
    """
    device = _get_device()
    model.eval()

    X_train = feat_scaler.transform(train_df[feature_cols].fillna(0))
    y_train = target_scaler.transform(train_df[TARGETS].fillna(0))
    X_test  = feat_scaler.transform(test_df[feature_cols].fillna(0))
    y_test  = target_scaler.transform(test_df[TARGETS].fillna(0))

    X_full = np.vstack([X_train, X_test])
    y_full = np.vstack([y_train, y_test])
    ds     = CurrentDataset(X_full, y_full, lookback)

    test_start = len(X_train) - lookback
    preds_sc   = []

    with torch.no_grad():
        for i in range(test_start, len(ds)):
            x, _ = ds[i]
            pred = model(x.unsqueeze(0).to(device)).cpu().numpy().squeeze()
            preds_sc.append(pred)

    preds_sc = np.array(preds_sc)
    preds    = target_scaler.inverse_transform(preds_sc)

    idx = test_df.index[:len(preds)]
    return (
        pd.Series(preds[:, 0], index=idx, name="lstm_resid_along"),
        pd.Series(preds[:, 1], index=idx, name="lstm_resid_cross"),
    )
