"""Main entry point: run RF stock classification strategy."""
import logging
import sys

from features import load_and_build_features
from model import run_walk_forward, plot_results

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def main():
    logger.info("=== Random Forest Stock Classification Strategy ===")
    logger.info("Target: 5-day return > 2%")
    logger.info("Features: price/volume technicals + PE/PB/PS + industry\n")

    # Load data and build features
    logger.info("Step 1: Loading data and computing features...")
    df, feature_cols, industry_names = load_and_build_features()
    logger.info(f"Features: {len(feature_cols)} columns")

    # Run walk-forward validation
    logger.info("\nStep 2: Running walk-forward validation...")
    results = run_walk_forward(df, feature_cols)

    # Summary
    logger.info("\n=== Summary Across All Windows ===")
    summary = results["summary"]
    logger.info(f"\n{summary.to_string(index=False)}")
    logger.info(f"\nMean metrics:")
    logger.info(f"  Accuracy:  {summary['accuracy'].mean():.4f}")
    logger.info(f"  Precision: {summary['precision'].mean():.4f}")
    logger.info(f"  Recall:    {summary['recall'].mean():.4f}")
    logger.info(f"  F1:        {summary['f1'].mean():.4f}")
    logger.info(f"  AUC:       {summary['auc'].mean():.4f}")

    # Feature importance
    logger.info("\n=== Top 10 Most Important Features ===")
    logger.info(results["importance"].head(10).to_string(index=False))

    # Generate plots
    logger.info("\nStep 3: Generating plots...")
    plot_results(results)

    logger.info("\n=== Done ===")
    logger.info("Check results/ folder for plots and detailed metrics.")


if __name__ == "__main__":
    main()
