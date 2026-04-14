"""Data washer pipeline: runs all validators on each stock file."""
import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from data_washer.validators import (
    validate_data_types,
    validate_duplicates,
    validate_ohlc_consistency,
    validate_zero_volume,
    validate_fundamentals,
    validate_price_spikes,
)

logger = logging.getLogger(__name__)

VALIDATORS = [
    validate_data_types,
    validate_duplicates,
    validate_ohlc_consistency,
    validate_zero_volume,
    validate_fundamentals,
    validate_price_spikes,
]


def wash_stock(code: str, raw_path: Path, cleaned_path: Path) -> dict:
    """Clean one stock file. Returns wash report dict."""
    df = pd.read_parquet(raw_path)
    report = {"code": code, "raw_rows": len(df), "issues": {}}

    for validator in VALIDATORS:
        df, issues = validator(df)
        report["issues"].update(issues)

    report["cleaned_rows"] = len(df)
    df.to_parquet(cleaned_path, index=False)
    return report


def wash_all(raw_dir: Path, cleaned_dir: Path, report_dir: Path) -> pd.DataFrame:
    """Run washer on all raw OHLCV files. Returns summary DataFrame."""
    cleaned_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    parquet_files = sorted(raw_dir.glob("*.parquet"))
    logger.info(f"Washing {len(parquet_files)} stock files...")

    reports = []
    for f in tqdm(parquet_files, desc="Washing"):
        code = f.stem
        cleaned_path = cleaned_dir / f.name
        try:
            report = wash_stock(code, f, cleaned_path)
            reports.append(report)
        except Exception as e:
            logger.error(f"Failed washing {code}: {e}")
            reports.append({"code": code, "error": str(e)})

    # Save detailed report as JSON
    report_path = report_dir / f"wash_report_{datetime.now():%Y%m%d}.json"
    with open(report_path, "w") as fp:
        json.dump(reports, fp, indent=2, default=str)
    logger.info(f"Wash report saved to {report_path}")

    # Summary statistics
    total = len(reports)
    errors = sum(1 for r in reports if "error" in r)
    ok_reports = [r for r in reports if "error" not in r]

    if ok_reports:
        total_issues = {
            "duplicates_removed": sum(r["issues"].get("duplicates_removed", 0) for r in ok_reports),
            "high_fixed": sum(r["issues"].get("high_fixed", 0) for r in ok_reports),
            "low_fixed": sum(r["issues"].get("low_fixed", 0) for r in ok_reports),
            "suspended_days": sum(r["issues"].get("suspended_days", 0) for r in ok_reports),
            "nans_filled": sum(r["issues"].get("nans_filled", 0) for r in ok_reports),
            "values_clipped": sum(r["issues"].get("values_clipped", 0) for r in ok_reports),
            "price_spikes_flagged": sum(r["issues"].get("price_spikes_flagged", 0) for r in ok_reports),
        }
    else:
        total_issues = {}

    logger.info(f"Wash complete: {total} stocks, {errors} errors")
    for k, v in total_issues.items():
        if v > 0:
            logger.info(f"  {k}: {v}")

    return pd.DataFrame(ok_reports)
