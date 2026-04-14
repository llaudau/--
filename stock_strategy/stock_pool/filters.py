"""Stock universe filters. Each returns (passed: bool, reason: str | None)."""
import re
from pathlib import Path

import pandas as pd

from stock_pool.config import (
    MIN_TRADING_DAYS, MIN_AVG_DAILY_AMOUNT,
    IPO_BUFFER_DAYS, MAX_SUSPENSION_RATIO,
)


def filter_st_stocks(code: str, basics_row: pd.Series, df: pd.DataFrame) -> tuple[bool, str | None]:
    """Exclude ST / *ST / PT stocks."""
    name = str(basics_row.get("name", ""))
    if re.search(r"\*?ST|PT", name):
        return False, f"ST/PT name: {name}"
    # Also check current is_st flag from cleaned OHLCV
    if not df.empty and int(df["is_st"].iloc[-1]) == 1:
        return False, "is_st flag=1"
    return True, None


def filter_short_history(df: pd.DataFrame) -> tuple[bool, str | None]:
    """Require at least MIN_TRADING_DAYS non-suspended trading days."""
    is_suspended = df.get("is_suspended", pd.Series(dtype=bool))
    active_days = (~is_suspended).sum() if len(is_suspended) > 0 else len(df)
    if active_days < MIN_TRADING_DAYS:
        return False, f"only {active_days} active trading days < {MIN_TRADING_DAYS}"
    return True, None


def filter_illiquid(df: pd.DataFrame) -> tuple[bool, str | None]:
    """Require average daily amount >= MIN_AVG_DAILY_AMOUNT over last 60 days."""
    recent = df.tail(60)
    if recent.empty or "amount" not in recent.columns:
        return False, "no amount data"
    avg_amount = recent["amount"].replace(0, pd.NA).mean()
    if pd.isna(avg_amount) or avg_amount < MIN_AVG_DAILY_AMOUNT:
        return False, f"avg daily amount {avg_amount:.0f} < {MIN_AVG_DAILY_AMOUNT:.0f}"
    return True, None


def filter_new_ipo(basics_row: pd.Series, df: pd.DataFrame) -> tuple[bool, str | None]:
    """Set valid_from = IPO date + IPO_BUFFER_DAYS trading days.
    Exclude if total trading days < IPO_BUFFER_DAYS.
    """
    ipo_date = basics_row.get("ipo_date", "")
    if not ipo_date:
        return True, None  # unknown IPO date, don't exclude
    try:
        ipo_dt = pd.to_datetime(ipo_date)
    except Exception:
        return True, None

    if df.empty:
        return False, "no data"
    first_date = df["date"].min()
    # Count trading days from IPO
    days_since_ipo = len(df[df["date"] >= ipo_dt])
    if days_since_ipo < IPO_BUFFER_DAYS:
        return False, f"only {days_since_ipo} days since IPO (buffer={IPO_BUFFER_DAYS})"
    return True, None


def filter_excessive_suspension(df: pd.DataFrame) -> tuple[bool, str | None]:
    """Exclude stocks with suspension ratio > MAX_SUSPENSION_RATIO."""
    if df.empty:
        return False, "no data"
    is_suspended = df.get("is_suspended", pd.Series([False] * len(df)))
    ratio = is_suspended.mean()
    if ratio > MAX_SUSPENSION_RATIO:
        return False, f"suspension ratio {ratio:.1%} > {MAX_SUSPENSION_RATIO:.0%}"
    return True, None
