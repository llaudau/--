"""Shared feature engineering: load cleaned data, compute features, create labels.

Moved from rf_strategy/features.py and parameterized so any model can use it.
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)

DEFAULT_LOOKBACKS = [5, 10, 20, 60]


def compute_stock_features(
    df: pd.DataFrame,
    lookbacks: list[int] = DEFAULT_LOOKBACKS,
    forward_days: int = 5,
    return_threshold: float = 0.02,
    min_date: str = "2022-01-01",
) -> pd.DataFrame:
    """Compute technical features for a single stock DataFrame."""
    df = df.sort_values("date").copy()
    df = df[df["date"] >= min_date]
    if len(df) < 120:
        return pd.DataFrame()

    close = df["close"]
    ret_1d = close.pct_change()

    # Momentum: past returns
    for lb in lookbacks:
        df[f"ret_{lb}d"] = close.pct_change(lb)

    # Volatility
    df["vol_20d"] = ret_1d.rolling(20).std()
    df["vol_60d"] = ret_1d.rolling(60).std()

    # Volume ratios
    df["volume_ratio_5d"] = df["volume"] / df["volume"].rolling(5).mean()
    df["volume_ratio_20d"] = df["volume"] / df["volume"].rolling(20).mean()

    # Price relative to moving averages
    for w in [5, 20, 60]:
        df[f"ma{w}_ratio"] = close / close.rolling(w).mean()

    # Price patterns
    df["high_low_range"] = (df["high"] - df["low"]) / close
    df["upper_shadow"] = (df["high"] - np.maximum(df["open"], close)) / close
    df["lower_shadow"] = (np.minimum(df["open"], close) - df["low"]) / close

    # Forward return and label
    df["fwd_ret"] = close.shift(-forward_days) / close - 1
    df["label"] = (df["fwd_ret"] > return_threshold).astype("int8")

    keep_cols = (
        ["date", "fwd_ret", "label",
         "turnover_rate", "pe_ttm", "pb_mrq", "ps_ttm", "pcf_ttm"]
        + [f"ret_{lb}d" for lb in lookbacks]
        + ["vol_20d", "vol_60d", "volume_ratio_5d", "volume_ratio_20d",
           "ma5_ratio", "ma20_ratio", "ma60_ratio",
           "high_low_range", "upper_shadow", "lower_shadow"]
    )
    return df[keep_cols]


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return list of feature column names (everything except meta/target cols)."""
    exclude = {"date", "bs_code", "code", "name", "fwd_ret", "label",
               "ipo_date", "out_date", "status", "industry_classification",
               "open", "high", "low", "close", "volume", "amount", "is_st",
               "is_suspended", "price_spike",
               "industry", "industry_code"}
    return [c for c in df.columns if c not in exclude]


def load_and_build_features(
    ohlcv_dir: Path,
    basics_path: Path,
    pool_codes: set[str] | None = None,
    forward_days: int = 5,
    return_threshold: float = 0.02,
    lookbacks: list[int] = DEFAULT_LOOKBACKS,
    min_date: str = "2022-01-01",
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Load all stock data, compute features, merge industry, return combined DataFrame.

    Args:
        ohlcv_dir: Directory with cleaned per-stock parquet files.
        basics_path: Path to the stock basics parquet.
        pool_codes: Set of stock codes in the pool (None = load all).
        forward_days: Prediction horizon in trading days.
        return_threshold: Label threshold.
        lookbacks: Lookback periods for momentum features.
        min_date: Only keep data from this date onward.

    Returns:
        (combined_df, feature_cols, industry_names)
    """
    # Load basics for industry mapping
    basics = pd.read_parquet(basics_path)
    industry_map = basics[["code", "industry"]].copy()
    industry_cats = industry_map["industry"].astype("category")
    industry_map["industry_code"] = industry_cats.cat.codes
    industry_names = industry_cats.cat.categories.tolist()

    parquet_files = sorted(ohlcv_dir.glob("*.parquet"))
    logger.info(f"Loading {len(parquet_files)} stock files...")

    all_dfs = []
    for f in tqdm(parquet_files, desc="Features"):
        code = f.stem
        if pool_codes is not None and code not in pool_codes:
            continue

        df = pd.read_parquet(f, columns=[
            "date", "open", "high", "low", "close", "volume",
            "turnover_rate", "pe_ttm", "pb_mrq", "ps_ttm", "pcf_ttm", "is_st",
        ])

        # Skip ST stocks (redundant with pool, but safe)
        if "is_st" in df.columns and len(df) > 0 and int(df["is_st"].iloc[-1]) == 1:
            continue

        feat = compute_stock_features(
            df, lookbacks=lookbacks, forward_days=forward_days,
            return_threshold=return_threshold, min_date=min_date,
        )
        if feat.empty:
            continue
        feat["code"] = code

        # Downcast floats to float32 to halve memory
        float_cols = feat.select_dtypes("float64").columns
        feat[float_cols] = feat[float_cols].astype("float32")
        all_dfs.append(feat)

    combined = pd.concat(all_dfs, ignore_index=True)
    logger.info(f"Combined shape before merge: {combined.shape}")

    # Merge industry as integer code
    combined = combined.merge(industry_map[["code", "industry_code"]], on="code", how="left")
    combined["industry_code"] = combined["industry_code"].fillna(-1).astype("int16")

    # Drop rows with NaN features or missing label
    feature_cols = get_feature_columns(combined)
    combined.dropna(subset=feature_cols + ["label"], inplace=True)
    combined.reset_index(drop=True, inplace=True)

    logger.info(f"Final dataset: {combined.shape}, label dist: {combined['label'].value_counts().to_dict()}")
    return combined, feature_cols, industry_names
