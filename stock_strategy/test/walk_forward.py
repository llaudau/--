"""Generic walk-forward validation engine. Works with any BaseModel."""
import logging

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, classification_report, f1_score,
    precision_score, recall_score, roc_auc_score,
)

from model.base import BaseModel
from test.config import MAX_TRAIN_SAMPLES

logger = logging.getLogger(__name__)


def run_walk_forward(
    model: BaseModel,
    df: pd.DataFrame,
    feature_cols: list[str],
    windows: list[tuple],
    forward_days: int,
) -> dict:
    """Run walk-forward validation across all windows with any BaseModel.

    Args:
        model: Any object implementing BaseModel interface.
        df: Combined feature DataFrame (date, code, label, feature_cols...).
        feature_cols: List of feature column names.
        windows: List of (train_start, train_end, test_start, test_end) tuples.
        forward_days: Prediction horizon (used for label leakage prevention).

    Returns:
        dict with keys:
          'summary'    — DataFrame of metrics per window
          'importance' — DataFrame of feature importance (from last window)
          'test_preds' — DataFrame of all test predictions
          'last_model' — trained model from the last window
    """
    results = []
    all_test_preds = []
    last_X_test = last_y_test = None

    for i, (train_start, train_end, test_start, test_end) in enumerate(windows):
        logger.info(f"\n=== Window {i+1}: Train {train_start}~{train_end}, Test {test_start}~{test_end} ===")

        # Label leakage prevention: exclude last FORWARD_DAYS trading days from training
        train_dates = sorted(
            df.loc[(df["date"] >= train_start) & (df["date"] <= train_end), "date"].unique()
        )
        if len(train_dates) > forward_days:
            safe_train_end = train_dates[-(forward_days + 1)]
        else:
            safe_train_end = train_dates[0] if train_dates else train_end

        train_mask = (df["date"] >= train_start) & (df["date"] <= safe_train_end)
        test_mask = (df["date"] >= test_start) & (df["date"] <= test_end)

        train_df = df.loc[train_mask]
        if len(train_df) > MAX_TRAIN_SAMPLES:
            train_df = train_df.sample(n=MAX_TRAIN_SAMPLES, random_state=42)
            logger.info(f"  Subsampled training to {MAX_TRAIN_SAMPLES}")

        X_train = train_df[feature_cols]
        y_train = train_df["label"]
        test_df = df.loc[test_mask]
        X_test = test_df[feature_cols]
        y_test = test_df["label"]

        if len(X_test) == 0:
            logger.warning(f"  No test data for window {i+1}, skipping")
            continue

        logger.info(f"  Train: {len(X_train)}, Test: {len(X_test)}")
        logger.info(f"  Train label dist: {np.bincount(y_train.values.astype(int))}")
        logger.info(f"  Test  label dist: {np.bincount(y_test.values.astype(int))}")

        # Train
        model.train(X_train, y_train)

        # Predict
        y_prob = model.predict_proba(X_test)
        y_pred = model.predict(X_test)

        # Metrics
        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec  = recall_score(y_test, y_pred, zero_division=0)
        f1   = f1_score(y_test, y_pred, zero_division=0)
        try:
            auc = roc_auc_score(y_test, y_prob)
        except ValueError:
            auc = float("nan")

        logger.info(f"  Acc={acc:.4f} Prec={prec:.4f} Rec={rec:.4f} F1={f1:.4f} AUC={auc:.4f}")
        logger.info(f"\n{classification_report(y_test, y_pred, target_names=['no','yes'], zero_division=0)}")

        results.append({
            "window": i + 1,
            "test_period": f"{test_start}~{test_end}",
            "train_size": len(X_train),
            "test_size": len(X_test),
            "accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "auc": auc,
        })

        # Accumulate test predictions for backtest
        preds_df = test_df[["date", "code", "fwd_ret", "label"]].copy()
        preds_df["y_prob"] = y_prob.values
        preds_df["y_pred"] = y_pred.values
        preds_df["window"] = i + 1
        all_test_preds.append(preds_df)

        last_X_test, last_y_test = X_test, y_test

    # Feature importance from last window
    imp_df = pd.DataFrame()
    if last_X_test is not None and last_y_test is not None:
        logger.info("\nComputing feature importance on last window...")
        try:
            imp_df = model.get_feature_importance(last_X_test, last_y_test, feature_cols)
        except Exception as e:
            logger.warning(f"Feature importance failed: {e}")

    summary = pd.DataFrame(results)
    test_preds = pd.concat(all_test_preds, ignore_index=True) if all_test_preds else pd.DataFrame()

    return {
        "summary": summary,
        "importance": imp_df,
        "test_preds": test_preds,
        "last_model": model,
    }
