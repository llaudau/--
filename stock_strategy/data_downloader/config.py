from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
OHLCV_DIR = DATA_DIR / "ohlcv"
BASICS_DIR = DATA_DIR / "basics"
META_DIR = DATA_DIR / "meta"
DOWNLOAD_LOG_PATH = META_DIR / "download_log.json"

# Download settings
OHLCV_HISTORY_START = "2010-01-01"  # how far back for initial full download
REQUEST_DELAY = 0.1  # seconds between baostock API calls (TCP, rate-limit friendly)
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds between retries

# Mainboard stock prefixes
# SZ: 000xxx, 001xxx, 002xxx, 003xxx
# SH: 600xxx, 601xxx, 603xxx, 605xxx
MAINBOARD_PREFIXES = ("000", "001", "002", "003", "600", "601", "603", "605")

# Baostock daily fields
# date,code,open,high,low,close,volume,amount,turn,peTTM,pbMRQ,psTTM,pcfNcfTTM,isST
BAOSTOCK_FIELDS = "date,code,open,high,low,close,volume,amount,turn,peTTM,pbMRQ,psTTM,pcfNcfTTM,isST"

BAOSTOCK_COLUMNS = {
    "date": "date",
    "code": "bs_code",
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "volume": "volume",
    "amount": "amount",
    "turn": "turnover_rate",
    "peTTM": "pe_ttm",
    "pbMRQ": "pb_mrq",
    "psTTM": "ps_ttm",
    "pcfNcfTTM": "pcf_ttm",
    "isST": "is_st",
}
