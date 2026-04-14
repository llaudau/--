"""Random Forest model: walk-forward validation, metrics, plots."""
import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, roc_auc_score, precision_score,
    recall_score, f1_score,
)
from sklearn.inspection import permutation_importance

from config import RF_PARAMS, WALK_FORWARD_WINDOWS, RESULTS_DIR

logger = logging.getLogger(__name__)


def run_walk_forward(df: pd.DataFrame, feature_cols: list[str]) -> dict:
    """Run walk-forward validation across all windows."""
    MAX_TRAIN_SAMPLES = 1_000_000  # cap training to avoid OOM
    results = []
    all_test_preds = []

    for i, (train_start, train_end, test_start, test_end) in enumerate(WALK_FORWARD_WINDOWS):
        logger.info(f"\n=== Window {i+1}: Train {train_start}~{train_end}, Test {test_start}~{test_end} ===")

        # Exclude last FORWARD_DAYS trading days from training to prevent
        # label leakage (labels use future prices that may fall into test period)
        from config import FORWARD_DAYS
        train_dates = sorted(df.loc[(df["date"] >= train_start) & (df["date"] <= train_end), "date"].unique())
        if len(train_dates) > FORWARD_DAYS:
            safe_train_end = train_dates[-(FORWARD_DAYS + 1)]
        else:
            safe_train_end = train_dates[0]

        train_mask = (df["date"] >= train_start) & (df["date"] <= safe_train_end)
        test_mask = (df["date"] >= test_start) & (df["date"] <= test_end)

        train_df = df.loc[train_mask]
        # Subsample training data if too large
        if len(train_df) > MAX_TRAIN_SAMPLES:
            train_df = train_df.sample(n=MAX_TRAIN_SAMPLES, random_state=42)
            logger.info(f"  Subsampled training to {MAX_TRAIN_SAMPLES}")

        X_train = train_df[feature_cols].values
        y_train = train_df["label"].values
        X_test = df.loc[test_mask, feature_cols].values
        y_test = df.loc[test_mask, "label"].values

        if len(X_test) == 0:
            logger.warning(f"  No test data for window {i+1}, skipping")
            continue

        logger.info(f"  Train: {len(X_train)} samples, Test: {len(X_test)} samples")
        logger.info(f"  Train label dist: {np.bincount(y_train.astype(int))}")
        logger.info(f"  Test  label dist: {np.bincount(y_test.astype(int))}")

        # Train RF
        clf = RandomForestClassifier(**RF_PARAMS)
        clf.fit(X_train, y_train)

        # Predict
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1]

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        try:
            auc = roc_auc_score(y_test, y_prob)
        except ValueError:
            auc = float("nan")

        logger.info(f"  Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
        logger.info(f"\n{classification_report(y_test, y_pred, target_names=['cant_earn', 'can_earn'], zero_division=0)}")

        window_result = {
            "window": i + 1,
            "test_period": f"{test_start}~{test_end}",
            "train_size": len(X_train),
            "test_size": len(X_test),
            "accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "auc": auc,
        }
        results.append(window_result)

        # Store test predictions for backtest
        test_df = df.loc[test_mask, ["date", "code", "fwd_ret", "label"]].copy()
        test_df["y_pred"] = y_pred
        test_df["y_prob"] = y_prob
        all_test_preds.append(test_df)

    # Feature importance from last window (subsample test for speed)
    logger.info("\nComputing permutation importance on last window...")
    n_imp = min(50000, len(X_test))
    imp_idx = np.random.RandomState(42).choice(len(X_test), n_imp, replace=False)
    perm_imp = permutation_importance(clf, X_test[imp_idx], y_test[imp_idx], n_repeats=5,
                                       random_state=42, n_jobs=-1)
    imp_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": perm_imp.importances_mean,
        "std": perm_imp.importances_std,
    }).sort_values("importance", ascending=False)

    results_summary = pd.DataFrame(results)
    test_preds = pd.concat(all_test_preds, ignore_index=True)

    return {
        "summary": results_summary,
        "importance": imp_df,
        "test_preds": test_preds,
        "last_clf": clf,
    }


def plot_results(results: dict):
    """Generate evaluation plots."""
    RESULTS_DIR.mkdir(exist_ok=True)
    summary = results["summary"]
    imp_df = results["importance"]
    test_preds = results["test_preds"]

    # 1. Metrics across walk-forward windows
    fig, ax = plt.subplots(figsize=(10, 5))
    x = summary["test_period"]
    ax.plot(x, summary["accuracy"], "o-", label="Accuracy")
    ax.plot(x, summary["precision"], "s-", label="Precision")
    ax.plot(x, summary["recall"], "^-", label="Recall")
    ax.plot(x, summary["f1"], "D-", label="F1")
    ax.plot(x, summary["auc"], "v-", label="AUC")
    ax.set_ylabel("Score")
    ax.set_title("Walk-Forward Validation Metrics")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1)
    plt.xticks(rotation=15)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "walk_forward_metrics.png", dpi=150)
    plt.close(fig)

    # 2. Feature importance (top 20)
    top_n = 20
    top_imp = imp_df.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(range(len(top_imp)), top_imp["importance"], xerr=top_imp["std"],
            color="steelblue", edgecolor="white")
    ax.set_yticks(range(len(top_imp)))
    ax.set_yticklabels(top_imp["feature"])
    ax.set_xlabel("Permutation Importance")
    ax.set_title(f"Top {top_n} Feature Importance (last window)")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "feature_importance.png", dpi=150)
    plt.close(fig)

    # 3. Simple backtest: weekly buy top predicted stocks vs random
    backtest_return(test_preds)

    logger.info(f"Plots saved to {RESULTS_DIR}")


def backtest_return(test_preds: pd.DataFrame):
    """Simple backtest: each week buy stocks RF predicts as 'can earn', measure actual return."""
    test_preds = test_preds.copy()
    test_preds["date"] = pd.to_datetime(test_preds["date"])

    # Group by week
    test_preds["week"] = test_preds["date"].dt.isocalendar().week.astype(int)
    test_preds["year"] = test_preds["date"].dt.year
    test_preds["yearweek"] = test_preds["year"] * 100 + test_preds["week"]

    weekly_groups = test_preds.groupby("yearweek")

    rf_returns = []
    random_returns = []
    all_returns = []

    for _, group in weekly_groups:
        if len(group) < 10:
            continue

        # RF picks: top 20 stocks by predicted probability
        top_picks = group.nlargest(20, "y_prob")
        rf_ret = top_picks["fwd_ret"].mean()

        # Random baseline: random 20 stocks
        random_pick = group.sample(n=min(20, len(group)), random_state=42)
        rand_ret = random_pick["fwd_ret"].mean()

        # Market average
        avg_ret = group["fwd_ret"].mean()

        rf_returns.append(rf_ret)
        random_returns.append(rand_ret)
        all_returns.append(avg_ret)

    # Cumulative returns
    rf_cum = (1 + pd.Series(rf_returns)).cumprod()
    rand_cum = (1 + pd.Series(random_returns)).cumprod()
    avg_cum = (1 + pd.Series(all_returns)).cumprod()

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(rf_cum.values, label=f"RF Top-20 (total: {rf_cum.iloc[-1]:.2f}x)", lw=2)
    ax.plot(rand_cum.values, label=f"Random-20 (total: {rand_cum.iloc[-1]:.2f}x)", lw=2, ls="--")
    ax.plot(avg_cum.values, label=f"Market Avg (total: {avg_cum.iloc[-1]:.2f}x)", lw=2, ls=":")
    ax.set_xlabel("Week")
    ax.set_ylabel("Cumulative Return")
    ax.set_title("Backtest: RF Top-20 Picks vs Random vs Market Average")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "backtest_cumulative.png", dpi=150)
    plt.close(fig)

    logger.info(f"\nBacktest Results:")
    logger.info(f"  RF Top-20 cumulative:  {rf_cum.iloc[-1]:.3f}x")
    logger.info(f"  Random-20 cumulative:  {rand_cum.iloc[-1]:.3f}x")
    logger.info(f"  Market Avg cumulative: {avg_cum.iloc[-1]:.3f}x")
