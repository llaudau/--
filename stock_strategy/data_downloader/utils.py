import json
import os
import time
import logging
import functools

import akshare as ak
import baostock as bs

from data_downloader.config import DOWNLOAD_LOG_PATH, META_DIR, MAX_RETRIES, RETRY_DELAY, MAINBOARD_PREFIXES

# Bypass proxy for Chinese financial APIs (East Money, baostock etc.)
for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY",
            "all_proxy", "ALL_PROXY"):
    os.environ.pop(key, None)


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(META_DIR / "download.log", encoding="utf-8"),
        ],
    )


def load_download_log() -> dict:
    if DOWNLOAD_LOG_PATH.exists():
        with open(DOWNLOAD_LOG_PATH, "r") as f:
            return json.load(f)
    return {}


def save_download_log(log: dict):
    DOWNLOAD_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(DOWNLOAD_LOG_PATH, "w") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)


def retry_on_failure(max_retries=MAX_RETRIES, delay=RETRY_DELAY):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt < max_retries - 1:
                        wait = delay * (2 ** attempt)
                        logging.warning(
                            f"{func.__name__} failed (attempt {attempt+1}/{max_retries}): {e}. "
                            f"Retrying in {wait}s..."
                        )
                        time.sleep(wait)
                    else:
                        logging.error(f"{func.__name__} failed after {max_retries} attempts: {e}")
                        raise
        return wrapper
    return decorator


def get_mainboard_stock_codes() -> list[str]:
    """Get all mainboard A-share stock codes via akshare."""
    df = ak.stock_info_a_code_name()
    codes = df[df["code"].str[:3].isin(MAINBOARD_PREFIXES)]["code"].tolist()
    logging.info(f"Got {len(codes)} mainboard stock codes")
    return sorted(codes)


def code_to_baostock(code: str) -> str:
    """Convert plain code '000001' to baostock format 'sz.000001'."""
    if code.startswith(("6",)):
        return f"sh.{code}"
    return f"sz.{code}"


def baostock_login():
    lg = bs.login()
    if lg.error_code != "0":
        raise RuntimeError(f"baostock login failed: {lg.error_msg}")
    return lg


def baostock_logout():
    bs.logout()
