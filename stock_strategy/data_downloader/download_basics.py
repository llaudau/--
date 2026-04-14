"""Download per-stock basic info (industry, listing date) via baostock."""
import time
import logging
from datetime import datetime

import baostock as bs
import pandas as pd
from tqdm import tqdm

from data_downloader.config import BASICS_DIR, REQUEST_DELAY
from data_downloader.utils import (
    get_mainboard_stock_codes,
    code_to_baostock,
    baostock_login,
    baostock_logout,
)


def download_stock_info(bs_code: str) -> dict | None:
    """Get basic info for one stock via baostock."""
    rs = bs.query_stock_basic(code=bs_code)
    data = rs.get_data()
    if data.empty:
        return None
    row = data.iloc[0]

    # Get industry
    rs2 = bs.query_stock_industry(code=bs_code)
    industry_data = rs2.get_data()
    industry = industry_data.iloc[0]["industry"] if not industry_data.empty else ""
    industry_class = industry_data.iloc[0]["industryClassification"] if not industry_data.empty else ""

    return {
        "code": bs_code.split(".")[1],
        "name": row.get("code_name", ""),
        "ipo_date": row.get("ipoDate", ""),
        "out_date": row.get("outDate", ""),
        "status": row.get("status", ""),
        "industry": industry,
        "industry_classification": industry_class,
    }


def download_all_basics():
    """Download basic info for all mainboard stocks via baostock. Saves incrementally."""
    codes = get_mainboard_stock_codes()
    today = datetime.now().strftime("%Y%m%d")
    path = BASICS_DIR / f"basics_{today}.parquet"

    baostock_login()
    records = []
    failed = 0

    try:
        for i, code in enumerate(tqdm(codes, desc="Basics")):
            bs_code = code_to_baostock(code)
            try:
                info = download_stock_info(bs_code)
                if info:
                    records.append(info)
            except Exception as e:
                logging.error(f"Basics failed {code}: {e}")
                failed += 1
            time.sleep(REQUEST_DELAY)

            # Save every 200 stocks for crash safety
            if (i + 1) % 200 == 0:
                df = pd.DataFrame(records)
                df.to_parquet(path, index=False)
                logging.info(f"Checkpoint: {len(df)} stocks saved")
    finally:
        baostock_logout()

    df = pd.DataFrame(records)
    df.to_parquet(path, index=False)
    logging.info(f"Saved basics: {len(df)} stocks, {failed} failed -> {path.name}")
    return df


if __name__ == "__main__":
    from utils import setup_logging
    setup_logging()
    download_all_basics()
