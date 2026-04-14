"""
Daily A-share mainboard stock data downloader.

Usage:
    python download_all.py              # incremental OHLCV+PE/PB update (daily use)
    python download_all.py --basics     # download stock basics (market cap, industry)
    python download_all.py --all        # both OHLCV and basics
"""
import argparse
import logging
from datetime import datetime

from data_downloader.config import BASICS_DIR
from data_downloader.utils import setup_logging
from data_downloader.download_daily_ohlcv import update_all_ohlcv
from data_downloader.download_basics import download_all_basics


def main():
    parser = argparse.ArgumentParser(description="Download A-share mainboard stock data")
    parser.add_argument("--basics", action="store_true", help="Download basics (market cap, industry)")
    parser.add_argument("--all", action="store_true", help="Download both OHLCV and basics")
    args = parser.parse_args()

    setup_logging()
    logging.info(f"=== Download started at {datetime.now():%Y-%m-%d %H:%M:%S} ===")

    # Default: OHLCV only (daily use)
    run_ohlcv = not args.basics or args.all
    run_basics = args.basics or args.all

    if run_ohlcv:
        logging.info("Updating OHLCV + PE/PB data (baostock)...")
        update_all_ohlcv()

    if run_basics:
        logging.info("Downloading basics (market cap, industry)...")
        download_all_basics()

    logging.info("=== Download complete ===")


if __name__ == "__main__":
    main()
