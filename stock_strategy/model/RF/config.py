"""RF-specific configuration — best parameters from risk-adjusted sweep."""

# Random Forest hyperparameters
RF_PARAMS = dict(
    n_estimators=300,
    max_features="sqrt",
    max_depth=8,
    min_samples_leaf=100,
    n_jobs=-1,
    random_state=42,
    class_weight="balanced",
)

# Best strategy parameters (risk-adjusted sweep winner)
# fwd=3d, thresh=2.0%, top_n=10 → cum=126x, Sharpe=4.10, MaxDD=-14.8%, SCORE=11.17
FORWARD_DAYS = 3
RETURN_THRESHOLD = 0.02
TOP_N = 10
MAX_TRAIN_SAMPLES = 1_000_000
