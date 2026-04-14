"""Stateless validation functions for stock OHLCV data.

Each validator takes a DataFrame and returns (cleaned_df, issues_dict).
issues_dict has keys like 'count', 'details' for the wash report.
"""
import numpy as np
import pandas as pd

from data_washer.config import (
    PRICE_SPIKE_THRESHOLD, PE_RANGE, PB_RANGE, PS_RANGE, PCF_RANGE,
)


def validate_data_types(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Ensure correct dtypes: is_st str->int8, date->datetime64, numerics."""
    issues = {"type_fixes": 0}
    df = df.copy()

    if df["date"].dtype != "datetime64[ns]" and df["date"].dtype != "datetime64[us]":
        df["date"] = pd.to_datetime(df["date"])
        issues["type_fixes"] += 1

    if df["is_st"].dtype == object or df["is_st"].dtype == "string":
        df["is_st"] = pd.to_numeric(df["is_st"], errors="coerce").fillna(0).astype("int8")
        issues["type_fixes"] += 1

    return df, issues


def validate_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Drop duplicate date rows, keeping last occurrence."""
    n_before = len(df)
    df = df.drop_duplicates(subset=["date"], keep="last")
    n_dropped = n_before - len(df)
    return df, {"duplicates_removed": n_dropped}


def validate_ohlc_consistency(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Fix OHLC: high must >= max(open,close), low must <= min(open,close)."""
    df = df.copy()
    oc_max = np.maximum(df["open"], df["close"])
    oc_min = np.minimum(df["open"], df["close"])

    high_fix = df["high"] < oc_max
    low_fix = df["low"] > oc_min

    n_high = high_fix.sum()
    n_low = low_fix.sum()

    df.loc[high_fix, "high"] = oc_max[high_fix]
    df.loc[low_fix, "low"] = oc_min[low_fix]

    return df, {"high_fixed": int(n_high), "low_fixed": int(n_low)}


def validate_zero_volume(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Flag rows with volume == 0 as suspended. Keep them but add is_suspended column."""
    df = df.copy()
    df["is_suspended"] = (df["volume"] == 0) | df["volume"].isna()
    n_suspended = df["is_suspended"].sum()
    return df, {"suspended_days": int(n_suspended)}


def validate_fundamentals(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Fill NaN fundamentals with ffill then bfill. Clip extreme values."""
    df = df.copy()
    fund_cols = ["turnover_rate", "pe_ttm", "pb_mrq", "ps_ttm", "pcf_ttm"]
    ranges = {
        "pe_ttm": PE_RANGE, "pb_mrq": PB_RANGE,
        "ps_ttm": PS_RANGE, "pcf_ttm": PCF_RANGE,
    }

    n_filled = 0
    n_clipped = 0

    for col in fund_cols:
        if col not in df.columns:
            continue
        nans_before = df[col].isna().sum()
        df[col] = df[col].ffill().bfill()
        n_filled += nans_before - df[col].isna().sum()

        if col in ranges:
            lo, hi = ranges[col]
            out_of_range = ((df[col] < lo) | (df[col] > hi)) & df[col].notna()
            n_clipped += out_of_range.sum()
            df[col] = df[col].clip(lo, hi)

    return df, {"nans_filled": int(n_filled), "values_clipped": int(n_clipped)}


def validate_price_spikes(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Flag single-day returns exceeding threshold. Does not remove rows."""
    df = df.copy()
    daily_ret = df["close"].pct_change().abs()
    spikes = daily_ret > PRICE_SPIKE_THRESHOLD
    df["price_spike"] = spikes.fillna(False)
    n_spikes = spikes.sum()
    return df, {"price_spikes_flagged": int(n_spikes)}
