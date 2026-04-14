"""Entry point: wash all raw OHLCV data and write cleaned files."""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import RAW_OHLCV_DIR, CLEANED_OHLCV_DIR, WASH_REPORTS_DIR
from data_washer.washer import wash_all

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

if __name__ == "__main__":
    wash_all(RAW_OHLCV_DIR, CLEANED_OHLCV_DIR, WASH_REPORTS_DIR)
