"""Systematic parameter sweep for RF stock strategy."""
import logging
import sys
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tqdm import tqdm

from config import OHLCV_DIR, BASICS_PATH, RESULTS_DIR, LOOKBACKS
from features import compute_stock_features, get_feature_columns

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

CACHE_PATH = RESULTS_DIR / "cached_features.parquet"
RESULTS_CSV = RESULTS_DIR / "param_sweep.csv"


def cache_features():
    """Load all stocks, compute features + multiple forward returns, save to disk."""
    if CACHE_PATH.exists():
        logger.info(f"Loading cached features from {CACHE_PATH}")
        return pd.read_parquet(CACHE_PATH)

    logger.info("Computing features from scratch...")
    basics = pd.read_parquet(BASICS_PATH)
    industry_map = basics[["code", "industry"]].copy()
    industry_cats = industry_map["industry"].astype("category")
    industry_map["industry_code"] = industry_cats.cat.codes

    parquet_files = sorted(OHLCV_DIR.glob("*.parquet"))
    all_dfs = []

    for f in tqdm(parquet_files, desc="Features"):
        code = f.stem
        df_raw = pd.read_parquet(f, columns=["date", "open", "high", "low", "close",
                                              "volume", "turnover_rate", "pe_ttm",
                                              "pb_mrq", "ps_ttm", "pcf_ttm", "is_st"])

        if "is_st" in df_raw.columns and df_raw["is_st"].iloc[-1] == 1:
            continue

        # Save close series before compute_stock_features drops it
        df_raw = df_raw.sort_values("date")
        df_raw["date"] = pd.to_datetime(df_raw["date"])
        close_raw = df_raw[df_raw["date"] >= "2022-01-01"].set_index("date")["close"].astype("float64")

        df = compute_stock_features(df_raw)
        if df.empty:
            continue

        # Compute forward returns for multiple horizons using original close
        df = df.set_index("date")
        for horizon in [3, 5, 10]:
            df[f"fwd_ret_{horizon}d"] = close_raw.shift(-horizon) / close_raw - 1
        df["code"] = code
        df.reset_index(inplace=True)

        # Downcast
        float_cols = df.select_dtypes("float64").columns
        df[float_cols] = df[float_cols].astype("float32")
        all_dfs.append(df)

    combined = pd.concat(all_dfs, ignore_index=True)
    combined = combined.merge(industry_map[["code", "industry_code"]], on="code", how="left")
    combined["industry_code"] = combined["industry_code"].fillna(-1).astype("int16")

    RESULTS_DIR.mkdir(exist_ok=True)
    combined.to_parquet(CACHE_PATH, index=False)
    logger.info(f"Cached {len(combined)} rows to {CACHE_PATH}")
    return combined


def make_labels(df: pd.DataFrame, fwd_days: int, threshold: float) -> pd.DataFrame:
    """Create binary labels from cached forward returns."""
    df = df.copy()
    fwd_col = f"fwd_ret_{fwd_days}d"
    df["fwd_ret"] = df[fwd_col]
    df["label"] = (df[fwd_col] > threshold).astype(int)

    # Get feature columns and exclude ALL forward return columns (data leak prevention)
    feature_cols = get_feature_columns(df)
    feature_cols = [c for c in feature_cols if not c.startswith("fwd_ret_")]

    # Drop rows with NaN in features or label
    df.dropna(subset=feature_cols + ["label"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df, feature_cols


def train_and_eval_single_window(df, feature_cols, rf_params, train_start, train_end, test_start, test_end,
                                  forward_days=3):
    """Train RF on one window, return metrics + test predictions.

    Excludes the last `forward_days` trading days from training to prevent
    label leakage (labels use future prices that fall into the test period).
    """
    # Get actual trading dates near train_end and exclude last forward_days
    train_dates = df.loc[(df["date"] >= train_start) & (df["date"] <= train_end), "date"].unique()
    train_dates = sorted(train_dates)
    if len(train_dates) > forward_days:
        safe_train_end = train_dates[-(forward_days + 1)]
    else:
        safe_train_end = train_dates[0]

    train_mask = (df["date"] >= train_start) & (df["date"] <= safe_train_end)
    test_mask = (df["date"] >= test_start) & (df["date"] <= test_end)

    train_df = df.loc[train_mask]
    if len(train_df) > 1_000_000:
        train_df = train_df.sample(n=1_000_000, random_state=42)

    X_train = train_df[feature_cols].values
    y_train = train_df["label"].values
    X_test = df.loc[test_mask, feature_cols].values
    y_test = df.loc[test_mask, "label"].values

    if len(X_test) == 0:
        return None

    clf = RandomForestClassifier(**rf_params)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }
    try:
        metrics["auc"] = roc_auc_score(y_test, y_prob)
    except ValueError:
        metrics["auc"] = float("nan")

    # Save test predictions for backtest
    test_df = df.loc[test_mask, ["date", "code", "fwd_ret"]].copy()
    test_df["y_prob"] = y_prob

    return metrics, test_df


def backtest_topn(test_preds: pd.DataFrame, top_n: int) -> float:
    """Compute cumulative return from weekly top-N picks."""
    test_preds = test_preds.copy()
    test_preds["week"] = test_preds["date"].dt.isocalendar().week
    test_preds["year"] = test_preds["date"].dt.year
    test_preds["yearweek"] = test_preds["year"] * 100 + test_preds["week"]

    weekly_returns = []
    for _, group in test_preds.groupby("yearweek"):
        if len(group) < top_n:
            continue
        top_picks = group.nlargest(top_n, "y_prob")
        weekly_returns.append(top_picks["fwd_ret"].mean())

    if not weekly_returns:
        return 1.0

    cumulative = (1 + pd.Series(weekly_returns)).prod()
    return cumulative


def run_sweep():
    """Run systematic parameter sweep."""
    logger.info("=== Parameter Sweep ===\n")

    # Step 1: Cache features
    df_cache = cache_features()

    # Define windows
    windows = [
        ("2022-01-01", "2023-06-30", "2023-07-01", "2023-12-31", "W1"),
        ("2022-01-01", "2023-12-31", "2024-01-01", "2024-12-31", "W2"),
        ("2022-01-01", "2024-12-31", "2025-01-01", "2025-12-31", "W3"),
    ]

    results = []

    # ===== GROUP A: RF Model Params =====
    logger.info("\n=== GROUP A: RF Model Params ===")
    logger.info("Screening on Window 2 (2024 test)...\n")

    max_depth_vals = [8, 12, 16]
    min_samples_leaf_vals = [30, 50, 100]

    # Use fixed strategy params for Group A
    df_a, feature_cols_a = make_labels(df_cache, fwd_days=5, threshold=0.02)

    screen_results_a = []
    for max_depth, min_samples_leaf in product(max_depth_vals, min_samples_leaf_vals):
        rf_params = {
            "n_estimators": 300,
            "max_features": "sqrt",
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
            "n_jobs": -1,
            "random_state": 42,
            "class_weight": "balanced",
        }

        logger.info(f"  Testing max_depth={max_depth}, min_samples_leaf={min_samples_leaf}")
        result = train_and_eval_single_window(df_a, feature_cols_a, rf_params, *windows[1][:4], forward_days=5)
        if result:
            metrics, _ = result
            screen_results_a.append({
                "max_depth": max_depth,
                "min_samples_leaf": min_samples_leaf,
                "auc": metrics["auc"],
                "rf_params": rf_params,
            })
            logger.info(f"    AUC: {metrics['auc']:.4f}")

    # Pick top 3 by AUC
    screen_results_a.sort(key=lambda x: x["auc"], reverse=True)
    top3_a = screen_results_a[:3]
    logger.info(f"\nTop 3 configs by AUC:")
    for i, cfg in enumerate(top3_a, 1):
        logger.info(f"  {i}. max_depth={cfg['max_depth']}, min_samples_leaf={cfg['min_samples_leaf']}, AUC={cfg['auc']:.4f}")

    # Validate top 3 on all windows
    logger.info("\nValidating top 3 on all windows...")
    for cfg in top3_a:
        for train_start, train_end, test_start, test_end, win_name in windows:
            result = train_and_eval_single_window(df_a, feature_cols_a, cfg["rf_params"],
                                                   train_start, train_end, test_start, test_end,
                                                   forward_days=5)
            if result:
                metrics, test_preds = result
                cum_ret = backtest_topn(test_preds, top_n=20)
                results.append({
                    "group": "A",
                    "max_depth": cfg["max_depth"],
                    "min_samples_leaf": cfg["min_samples_leaf"],
                    "forward_days": 5,
                    "return_threshold": 0.02,
                    "top_n": 20,
                    "window": win_name,
                    **metrics,
                    "cumulative_return": cum_ret,
                })

    # Pick best from Group A
    df_results_a = pd.DataFrame([r for r in results if r["group"] == "A"])
    best_a = df_results_a.groupby(["max_depth", "min_samples_leaf"])["auc"].mean().idxmax()
    logger.info(f"\nBest Group A: max_depth={best_a[0]}, min_samples_leaf={best_a[1]}")

    # ===== GROUP B: Strategy Params =====
    logger.info("\n=== GROUP B: Strategy Params ===")
    logger.info("Screening on Window 2 (2024 test)...\n")

    forward_days_vals = [3, 5, 10]
    return_threshold_vals = [0.01, 0.02, 0.03]

    best_rf_params = {
        "n_estimators": 300,
        "max_features": "sqrt",
        "max_depth": best_a[0],
        "min_samples_leaf": best_a[1],
        "n_jobs": -1,
        "random_state": 42,
        "class_weight": "balanced",
    }

    screen_results_b = []
    for fwd_days, threshold in product(forward_days_vals, return_threshold_vals):
        logger.info(f"  Testing forward_days={fwd_days}, return_threshold={threshold}")
        df_b, feature_cols_b = make_labels(df_cache, fwd_days=fwd_days, threshold=threshold)
        result = train_and_eval_single_window(df_b, feature_cols_b, best_rf_params, *windows[1][:4],
                                               forward_days=fwd_days)
        if result:
            metrics, _ = result
            screen_results_b.append({
                "forward_days": fwd_days,
                "return_threshold": threshold,
                "auc": metrics["auc"],
            })
            logger.info(f"    AUC: {metrics['auc']:.4f}")

    # Pick top 3 by AUC
    screen_results_b.sort(key=lambda x: x["auc"], reverse=True)
    top3_b = screen_results_b[:3]
    logger.info(f"\nTop 3 configs by AUC:")
    for i, cfg in enumerate(top3_b, 1):
        logger.info(f"  {i}. forward_days={cfg['forward_days']}, return_threshold={cfg['return_threshold']}, AUC={cfg['auc']:.4f}")

    # Validate top 3 on all windows
    logger.info("\nValidating top 3 on all windows...")
    for cfg in top3_b:
        df_b, feature_cols_b = make_labels(df_cache, fwd_days=cfg["forward_days"], threshold=cfg["return_threshold"])
        for train_start, train_end, test_start, test_end, win_name in windows:
            result = train_and_eval_single_window(df_b, feature_cols_b, best_rf_params,
                                                   train_start, train_end, test_start, test_end,
                                                   forward_days=cfg["forward_days"])
            if result:
                metrics, test_preds = result
                cum_ret = backtest_topn(test_preds, top_n=20)
                results.append({
                    "group": "B",
                    "max_depth": best_a[0],
                    "min_samples_leaf": best_a[1],
                    "forward_days": cfg["forward_days"],
                    "return_threshold": cfg["return_threshold"],
                    "top_n": 20,
                    "window": win_name,
                    **metrics,
                    "cumulative_return": cum_ret,
                })

    # Pick best from Group B
    df_results_b = pd.DataFrame([r for r in results if r["group"] == "B"])
    best_b = df_results_b.groupby(["forward_days", "return_threshold"])["cumulative_return"].mean().idxmax()
    logger.info(f"\nBest Group B: forward_days={best_b[0]}, return_threshold={best_b[1]}")

    # ===== GROUP C: Portfolio Params (Top-N) =====
    logger.info("\n=== GROUP C: Portfolio Params (Top-N) ===")
    logger.info("Re-ranking saved predictions with different Top-N values...\n")

    # Use best config from A+B, retrain to get predictions
    df_c, feature_cols_c = make_labels(df_cache, fwd_days=best_b[0], threshold=best_b[1])
    all_test_preds = []
    for train_start, train_end, test_start, test_end, win_name in windows:
        result = train_and_eval_single_window(df_c, feature_cols_c, best_rf_params,
                                               train_start, train_end, test_start, test_end,
                                               forward_days=best_b[0])
        if result:
            metrics, test_preds = result
            test_preds["window"] = win_name
            all_test_preds.append(test_preds)

    combined_preds = pd.concat(all_test_preds, ignore_index=True)

    top_n_vals = [5, 10, 20, 50]
    for top_n in top_n_vals:
        logger.info(f"  Testing top_n={top_n}")
        for win_name in ["W1", "W2", "W3"]:
            win_preds = combined_preds[combined_preds["window"] == win_name]
            cum_ret = backtest_topn(win_preds, top_n=top_n)
            results.append({
                "group": "C",
                "max_depth": best_a[0],
                "min_samples_leaf": best_a[1],
                "forward_days": best_b[0],
                "return_threshold": best_b[1],
                "top_n": top_n,
                "window": win_name,
                "accuracy": float("nan"),
                "precision": float("nan"),
                "recall": float("nan"),
                "f1": float("nan"),
                "auc": float("nan"),
                "cumulative_return": cum_ret,
            })
            logger.info(f"    {win_name}: {cum_ret:.3f}x")

    # Save all results
    df_results = pd.DataFrame(results)
    df_results.to_csv(RESULTS_CSV, index=False)
    logger.info(f"\n=== Results saved to {RESULTS_CSV} ===")

    # Summary
    logger.info("\n=== SUMMARY ===")
    logger.info(f"\nBest Group A (RF params):")
    logger.info(f"  max_depth={best_a[0]}, min_samples_leaf={best_a[1]}")
    logger.info(f"  Mean AUC: {df_results_a.groupby(['max_depth', 'min_samples_leaf'])['auc'].mean()[best_a]:.4f}")

    logger.info(f"\nBest Group B (strategy params):")
    logger.info(f"  forward_days={best_b[0]}, return_threshold={best_b[1]}")
    logger.info(f"  Mean cumulative return: {df_results_b.groupby(['forward_days', 'return_threshold'])['cumulative_return'].mean()[best_b]:.3f}x")

    df_results_c = pd.DataFrame([r for r in results if r["group"] == "C"])
    best_c = df_results_c.groupby("top_n")["cumulative_return"].mean().idxmax()
    logger.info(f"\nBest Group C (portfolio size):")
    logger.info(f"  top_n={best_c}")
    logger.info(f"  Mean cumulative return: {df_results_c.groupby('top_n')['cumulative_return'].mean()[best_c]:.3f}x")

    logger.info(f"\n=== Overall Best Config ===")
    logger.info(f"  max_depth={best_a[0]}, min_samples_leaf={best_a[1]}")
    logger.info(f"  forward_days={best_b[0]}, return_threshold={best_b[1]}")
    logger.info(f"  top_n={best_c}")


if __name__ == "__main__":
    run_sweep()
