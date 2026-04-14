"""Build and persist the stock pool by applying all filters."""
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from stock_pool.filters import (
    filter_st_stocks,
    filter_short_history,
    filter_illiquid,
    filter_new_ipo,
    filter_excessive_suspension,
)

logger = logging.getLogger(__name__)


def build_pool(basics_path: Path, cleaned_ohlcv_dir: Path, pool_dir: Path) -> pd.DataFrame:
    """Apply all filters and save the stock pool parquet.

    Returns the full DataFrame with reason_excluded column
    (None means the stock passed all filters).
    """
    pool_dir.mkdir(parents=True, exist_ok=True)
    basics = pd.read_parquet(basics_path)
    logger.info(f"Building stock pool from {len(basics)} stocks...")

    FILTERS = [
        ("st",         lambda code, row, df: filter_st_stocks(code, row, df)),
        ("history",    lambda code, row, df: filter_short_history(df)),
        ("liquidity",  lambda code, row, df: filter_illiquid(df)),
        ("ipo_buffer", lambda code, row, df: filter_new_ipo(row, df)),
        ("suspension", lambda code, row, df: filter_excessive_suspension(df)),
    ]

    records = []
    filter_counts = {name: 0 for name, _ in FILTERS}

    for _, row in tqdm(basics.iterrows(), total=len(basics), desc="Filtering"):
        code = row["code"]
        parquet = cleaned_ohlcv_dir / f"{code}.parquet"

        # Load cleaned OHLCV for this stock
        if parquet.exists():
            try:
                df = pd.read_parquet(parquet)
            except Exception:
                df = pd.DataFrame()
        else:
            df = pd.DataFrame()

        reason = None
        for filter_name, filter_fn in FILTERS:
            passed, fail_reason = filter_fn(code, row, df)
            if not passed:
                reason = f"[{filter_name}] {fail_reason}"
                filter_counts[filter_name] += 1
                break

        records.append({
            "code": code,
            "name": row.get("name", ""),
            "industry": row.get("industry", ""),
            "ipo_date": row.get("ipo_date", ""),
            "reason_excluded": reason,
        })

    pool_df = pd.DataFrame(records)
    passed = pool_df["reason_excluded"].isna().sum()
    total = len(pool_df)

    logger.info(f"Stock pool: {passed}/{total} stocks passed all filters")
    for fname, count in filter_counts.items():
        if count > 0:
            logger.info(f"  Excluded by [{fname}]: {count}")

    # Save with today's date stamp
    out_path = pool_dir / f"stock_pool_{datetime.now():%Y%m%d}.parquet"
    pool_df.to_parquet(out_path, index=False)

    # Also save a canonical "latest" copy for easy loading
    latest_path = pool_dir / "stock_pool_latest.parquet"
    pool_df.to_parquet(latest_path, index=False)
    logger.info(f"Pool saved to {out_path}")
    return pool_df


def load_pool(pool_dir: Path) -> pd.DataFrame:
    """Load the latest stock pool, returning only stocks that passed filters."""
    latest = pool_dir / "stock_pool_latest.parquet"
    if not latest.exists():
        raise FileNotFoundError(f"No stock pool found. Run: python -m stock_pool.run")
    df = pd.read_parquet(latest)
    return df[df["reason_excluded"].isna()].reset_index(drop=True)
