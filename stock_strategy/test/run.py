"""Entry point: run walk-forward validation + backtest + report for a selected model.

Usage:
    python -m test.run --model RF
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import CLEANED_OHLCV_DIR, RAW_BASICS_DIR, POOL_DIR, RESULTS_DIR

from model.features import load_and_build_features
from stock_pool.pool_builder import load_pool
from test.walk_forward import run_walk_forward
from test.backtest import backtest_return
from test.report import plot_all

# ── Model Registry ───────────────────────────────────────────────────────────
# To add a new model, create model/XGB/xgb_model.py implementing BaseModel,
# then add one entry here:
#   "XGB": ("model.XGB.xgb_model", "XGBModel", "model.XGB.config"),
MODEL_REGISTRY = {
    "RF": ("model.RF.rf_model", "RFModel", "model.RF.config"),
}


def _load_model(model_name: str):
    """Dynamically import and instantiate a model from the registry."""
    mod_path, cls_name, cfg_path = MODEL_REGISTRY[model_name]

    import importlib
    mod = importlib.import_module(mod_path)
    ModelClass = getattr(mod, cls_name)
    cfg = importlib.import_module(cfg_path)

    config = {
        "rf_params": getattr(cfg, "RF_PARAMS", {}),
        "forward_days": getattr(cfg, "FORWARD_DAYS", 5),
        "return_threshold": getattr(cfg, "RETURN_THRESHOLD", 0.02),
    }
    return ModelClass(config), config


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run backtest with a selected model")
    parser.add_argument("--model", default="RF", choices=MODEL_REGISTRY.keys())
    args = parser.parse_args()

    logger.info(f"=== Backtest Pipeline: {args.model} ===\n")

    # Load model + config
    model, model_config = _load_model(args.model)
    forward_days = model_config["forward_days"]
    return_threshold = model_config["return_threshold"]
    cfg_mod = __import__(MODEL_REGISTRY[args.model][2], fromlist=[""])
    top_n = getattr(cfg_mod, "TOP_N", 20)

    # Load stock pool
    pool_df = load_pool(POOL_DIR)
    pool_codes = set(pool_df["code"].tolist())
    logger.info(f"Stock pool: {len(pool_codes)} stocks")

    # Find basics
    basics_files = sorted(RAW_BASICS_DIR.glob("basics_*.parquet"))
    basics_path = basics_files[-1]

    # Build features
    logger.info("Loading features...")
    df, feature_cols, _ = load_and_build_features(
        ohlcv_dir=CLEANED_OHLCV_DIR,
        basics_path=basics_path,
        pool_codes=pool_codes,
        forward_days=forward_days,
        return_threshold=return_threshold,
    )
    logger.info(f"Features: {len(feature_cols)} columns, {len(df)} rows")

    # Walk-forward
    from test.config import WALK_FORWARD_WINDOWS
    results = run_walk_forward(model, df, feature_cols, WALK_FORWARD_WINDOWS, forward_days)

    # Summary
    summary = results["summary"]
    logger.info(f"\n{summary.to_string(index=False)}")
    logger.info(f"\nMean metrics:")
    for m in ["accuracy", "precision", "recall", "f1", "auc"]:
        logger.info(f"  {m.capitalize():>10}: {summary[m].mean():.4f}")

    # Feature importance
    if not results["importance"].empty:
        logger.info(f"\nTop 10 features:\n{results['importance'].head(10).to_string(index=False)}")

    # Backtest
    logger.info(f"\nRunning backtest with top_n={top_n}...")
    bt = backtest_return(results["test_preds"], top_n=top_n)

    # Report
    logger.info("Generating report...")
    model_results_dir = RESULTS_DIR / args.model
    plot_all(results, bt, top_n, model_results_dir)

    logger.info(f"\n=== Done. Results in {model_results_dir} ===")


if __name__ == "__main__":
    main()
