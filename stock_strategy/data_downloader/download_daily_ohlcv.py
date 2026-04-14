import time
import logging
from datetime import datetime, timedelta

import baostock as bs
import pandas as pd
from tqdm import tqdm

from data_downloader.config import (
    OHLCV_DIR,
    OHLCV_HISTORY_START,
    BAOSTOCK_FIELDS,
    BAOSTOCK_COLUMNS,
    REQUEST_DELAY,
)
from data_downloader.utils import (
    load_download_log,
    save_download_log,
    get_mainboard_stock_codes,
    code_to_baostock,
    baostock_login,
    baostock_logout,
    retry_on_failure,
)


def download_ohlcv_for_stock(bs_code: str, start_date: str, end_date: str) -> pd.DataFrame | None:
    """Download daily OHLCV+PE/PB for one stock via baostock. Returns None if no data."""
    rs = bs.query_history_k_data_plus(
        bs_code, BAOSTOCK_FIELDS,
        start_date=start_date, end_date=end_date,
        frequency="d", adjustflag="2",  # forward-adjusted
    )
    rows = []
    while rs.next():
        rows.append(rs.get_row_data())
    if not rows:
        return None

    df = pd.DataFrame(rows, columns=rs.fields)
    df.rename(columns=BAOSTOCK_COLUMNS, inplace=True)
    df["date"] = pd.to_datetime(df["date"])

    # Convert numeric columns
    numeric_cols = ["open", "high", "low", "close", "volume", "amount",
                    "turnover_rate", "pe_ttm", "pb_mrq", "ps_ttm", "pcf_ttm"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def update_all_ohlcv():
    """Incremental OHLCV+PE/PB update for all mainboard stocks."""
    log = load_download_log()
    all_codes = get_mainboard_stock_codes()
    today = datetime.now().strftime("%Y-%m-%d")

    baostock_login()

    updated = 0
    skipped = 0
    failed = 0

    try:
        for code in tqdm(all_codes, desc="Downloading"):
            bs_code = code_to_baostock(code)
            last_date = log.get(code)

            if last_date:
                start = (datetime.strptime(last_date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
                if start > today:
                    skipped += 1
                    continue
            else:
                start = OHLCV_HISTORY_START

            try:
                df_new = download_ohlcv_for_stock(bs_code, start, today)
                parquet_path = OHLCV_DIR / f"{code}.parquet"

                if df_new is not None and not df_new.empty:
                    if parquet_path.exists():
                        df_old = pd.read_parquet(parquet_path)
                        df_combined = pd.concat([df_old, df_new]).drop_duplicates(subset=["date"])
                        df_combined.sort_values("date", inplace=True)
                        df_combined.reset_index(drop=True, inplace=True)
                    else:
                        df_combined = df_new

                    df_combined.to_parquet(parquet_path, index=False)
                    updated += 1

                    # Store actual last trading date from data, not today's date
                    last_data_date = df_new["date"].max()
                    if hasattr(last_data_date, "strftime"):
                        log[code] = last_data_date.strftime("%Y-%m-%d")
                    else:
                        log[code] = str(last_data_date)[:10]
                    save_download_log(log)

            except Exception as e:
                logging.error(f"Failed {code}: {e}")
                failed += 1

            time.sleep(REQUEST_DELAY)
    finally:
        baostock_logout()

    logging.info(f"Done: {updated} updated, {skipped} up-to-date, {failed} failed")


if __name__ == "__main__":
    from utils import setup_logging
    setup_logging()
    update_all_ohlcv()
