"""Project-level shared configuration — paths only."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"

# Data pipeline directories
RAW_DIR = DATA_DIR / "raw"
RAW_OHLCV_DIR = RAW_DIR / "ohlcv"
RAW_BASICS_DIR = RAW_DIR / "basics"
RAW_META_DIR = RAW_DIR / "meta"

CLEANED_DIR = DATA_DIR / "cleaned"
CLEANED_OHLCV_DIR = CLEANED_DIR / "ohlcv"
WASH_REPORTS_DIR = CLEANED_DIR / "reports"

POOL_DIR = DATA_DIR / "pool"
MODELS_DIR = DATA_DIR / "models"
RESULTS_DIR = DATA_DIR / "results"
PREDICTIONS_DIR = DATA_DIR / "predictions"
