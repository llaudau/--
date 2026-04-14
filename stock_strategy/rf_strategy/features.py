"""Feature engineering: load raw data, compute features, create labels."""
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from config import (
    OHLCV_DIR, BASICS_PATH, FORWARD_DAYS, RETURN_THRESHOLD, LOOKBACKS,
)

logger = logging.getLogger(__name__)


def compute_stock_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute technical features for a single stock DataFrame."""
    df = df.sort_values("date").copy()
    # Only keep rows from 2022 onwards (pre-2022 macro env too different)
    df = df[df["date"] >= "2022-01-01"]
    if len(df) < 120:
        return pd.DataFrame()
    close = df["close"]
    ret_1d = close.pct_change()

    # Momentum: past returns
    for lb in LOOKBACKS:
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

    # Target: 5-day forward return > 2%
    df["fwd_ret"] = close.shift(-FORWARD_DAYS) / close - 1
    df["label"] = (df["fwd_ret"] > RETURN_THRESHOLD).astype(int)

    # Only keep feature columns + meta to reduce memory
    keep_cols = (["date", "fwd_ret", "label",
                  "turnover_rate", "pe_ttm", "pb_mrq", "ps_ttm", "pcf_ttm"]
                 + [f"ret_{lb}d" for lb in LOOKBACKS]
                 + ["vol_20d", "vol_60d", "volume_ratio_5d", "volume_ratio_20d",
                    "ma5_ratio", "ma20_ratio", "ma60_ratio",
                    "high_low_range", "upper_shadow", "lower_shadow"])
    return df[keep_cols]


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return list of feature column names (everything except meta/target cols)."""
    exclude = {"date", "bs_code", "code", "name", "fwd_ret", "label",
               "ipo_date", "out_date", "status", "industry_classification",
               "open", "high", "low", "close", "volume", "amount", "is_st",
               "industry", "industry_code"}
    return [c for c in df.columns if c not in exclude]


def load_and_build_features() -> pd.DataFrame:
    """Load all stock data, compute features, merge industry, return combined DataFrame."""
    # Load basics for industry mapping — encode as integer to save memory
    basics = pd.read_parquet(BASICS_PATH)
    industry_map = basics[["code", "industry"]].copy()
    industry_cats = industry_map["industry"].astype("category")
    industry_map["industry_code"] = industry_cats.cat.codes
    industry_names = industry_cats.cat.categories.tolist()

    # Load all stock parquet files and compute features
    parquet_files = sorted(OHLCV_DIR.glob("*.parquet"))
    logger.info(f"Loading {len(parquet_files)} stock files...")

    all_dfs = []
    for f in tqdm(parquet_files, desc="Features"):
        code = f.stem
        df = pd.read_parquet(f, columns=["date", "open", "high", "low", "close",
                                          "volume", "turnover_rate", "pe_ttm",
                                          "pb_mrq", "ps_ttm", "pcf_ttm", "is_st"])

        # Skip ST stocks
        if "is_st" in df.columns and df["is_st"].iloc[-1] == 1:
            continue

        df = compute_stock_features(df)
        if df.empty:
            continue
        df["code"] = code

        # Downcast floats to float32 to halve memory
        float_cols = df.select_dtypes("float64").columns
        df[float_cols] = df[float_cols].astype("float32")

        all_dfs.append(df)

    combined = pd.concat(all_dfs, ignore_index=True)
    logger.info(f"Combined shape before merge: {combined.shape}")

    # Merge industry as integer code (much lighter than one-hot)
    combined = combined.merge(industry_map[["code", "industry_code"]], on="code", how="left")
    combined["industry_code"] = combined["industry_code"].fillna(-1).astype("int16")

    # Drop rows with NaN features or missing label
    feature_cols = get_feature_columns(combined)
    combined.dropna(subset=feature_cols + ["label"], inplace=True)
    combined.reset_index(drop=True, inplace=True)

    logger.info(f"Final dataset: {combined.shape}, label dist: {combined['label'].value_counts().to_dict()}")
    logger.info(f"Memory usage: {combined.memory_usage(deep=True).sum() / 1e9:.2f} GB")
    return combined, feature_cols, industry_names
