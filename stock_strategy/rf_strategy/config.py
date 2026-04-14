from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "results"
DATA_DIR = BASE_DIR.parent / "data_downloader" / "data"
OHLCV_DIR = DATA_DIR / "ohlcv"
BASICS_PATH = DATA_DIR / "basics" / "basics_20260410.parquet"

# Strategy parameters
FORWARD_DAYS = 5          # prediction horizon
RETURN_THRESHOLD = 0.02   # >2% = "can earn"

# Feature lookback periods
LOOKBACKS = [5, 10, 20, 60]

# Walk-forward windows: (train_start, train_end, test_start, test_end)
WALK_FORWARD_WINDOWS = [
    ("2022-01-01", "2023-06-30", "2023-07-01", "2023-12-31"),
    ("2022-01-01", "2023-12-31", "2024-01-01", "2024-12-31"),
    ("2022-01-01", "2024-12-31", "2025-01-01", "2025-12-31"),
]

# Random Forest parameters
RF_PARAMS = dict(
    n_estimators=300,
    max_features="sqrt",
    max_depth=12,
    min_samples_leaf=50,
    n_jobs=-1,
    random_state=42,
    class_weight="balanced",  # handle class imbalance
)

# Feature columns (filled dynamically by features.py)
FEATURE_COLS = []
