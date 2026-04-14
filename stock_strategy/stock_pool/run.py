"""Entry point: build the stock pool from cleaned data + basics."""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import CLEANED_OHLCV_DIR, POOL_DIR, RAW_BASICS_DIR
from stock_pool.pool_builder import build_pool

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

if __name__ == "__main__":
    # Find the latest basics file
    basics_files = sorted(RAW_BASICS_DIR.glob("basics_*.parquet"))
    if not basics_files:
        raise FileNotFoundError(f"No basics parquet in {RAW_BASICS_DIR}")
    basics_path = basics_files[-1]
    build_pool(basics_path, CLEANED_OHLCV_DIR, POOL_DIR)
