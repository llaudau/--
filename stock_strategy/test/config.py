"""Test module configuration — walk-forward windows and backtest settings."""

WALK_FORWARD_WINDOWS = [
    ("2022-01-01", "2023-06-30", "2023-07-01", "2023-12-31"),
    ("2022-01-01", "2023-12-31", "2024-01-01", "2024-12-31"),
    ("2022-01-01", "2024-12-31", "2025-01-01", "2025-12-31"),
]

BACKTEST_TOP_N = 5
MAX_TRAIN_SAMPLES = 1_000_000
