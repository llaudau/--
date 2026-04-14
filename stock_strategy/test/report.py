"""Report generation: plots + text summary + overfitting diagnostics."""
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger(__name__)


def plot_all(results: dict, backtest: dict, top_n: int, results_dir: Path) -> None:
    """Generate all plots and text report. Save to results_dir."""
    results_dir.mkdir(parents=True, exist_ok=True)
    summary = results["summary"]
    imp_df = results["importance"]

    _plot_walk_forward_metrics(summary, results_dir)
    _plot_feature_importance(imp_df, results_dir)
    _plot_backtest_cumulative(backtest, top_n, results_dir)
    _plot_overfit_diagnostic(summary, results_dir)
    _plot_return_stability(backtest, results_dir)
    _write_text_report(summary, backtest, top_n, results_dir)
    logger.info(f"All outputs saved to {results_dir}")


def _plot_walk_forward_metrics(summary: pd.DataFrame, d: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    x = summary["test_period"]
    for col, marker in [("accuracy","o"), ("precision","s"), ("recall","^"),
                         ("f1","D"), ("auc","v")]:
        ax.plot(x, summary[col], f"{marker}-", label=col.capitalize())
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Walk-Forward Validation Metrics")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.xticks(rotation=15)
    fig.tight_layout()
    fig.savefig(d / "walk_forward_metrics.png", dpi=150)
    plt.close(fig)


def _plot_feature_importance(imp_df: pd.DataFrame, d: Path) -> None:
    if imp_df.empty:
        return
    top = imp_df.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(range(len(top)), top["importance"], xerr=top["std"],
            color="steelblue", edgecolor="white")
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["feature"])
    ax.set_xlabel("Permutation Importance")
    ax.set_title("Top 20 Feature Importance (last window)")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(d / "feature_importance.png", dpi=150)
    plt.close(fig)


def _plot_backtest_cumulative(bt: dict, top_n: int, d: Path) -> None:
    rf_cum = bt["rf_cum"]
    rand_cum = bt["rand_cum"]
    avg_cum = bt["avg_cum"]
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(rf_cum.values,   label=f"RF Top-{top_n} ({rf_cum.iloc[-1]:.2f}x)", lw=2)
    ax.plot(rand_cum.values, label=f"Random-{top_n} ({rand_cum.iloc[-1]:.2f}x)", lw=2, ls="--")
    ax.plot(avg_cum.values,  label=f"Market Avg ({avg_cum.iloc[-1]:.2f}x)", lw=2, ls=":")
    ax.set_xlabel("Week")
    ax.set_ylabel("Cumulative Return")
    ax.set_title(f"Backtest: RF Top-{top_n} vs Random vs Market")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(d / "backtest_cumulative.png", dpi=150)
    plt.close(fig)


def _plot_overfit_diagnostic(summary: pd.DataFrame, d: Path) -> None:
    """Show AUC per window to detect decay (sign of overfitting)."""
    if summary.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    x = summary["test_period"]
    ax.bar(x, summary["auc"], color=["steelblue" if v >= 0.55 else "tomato"
                                      for v in summary["auc"]])
    ax.axhline(0.5, color="red", ls="--", lw=1, label="Random baseline (0.50)")
    ax.set_ylim(0.4, 0.8)
    ax.set_ylabel("AUC")
    ax.set_title("Overfitting Check: AUC per Test Window")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    plt.xticks(rotation=15)
    fig.tight_layout()
    fig.savefig(d / "overfit_diagnostic.png", dpi=150)
    plt.close(fig)


def _plot_return_stability(bt: dict, d: Path) -> None:
    """Box plot of weekly RF returns to visualise consistency."""
    rf_weekly = bt["rf_weekly"]
    if not rf_weekly:
        return
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.boxplot(rf_weekly, vert=True, patch_artist=True,
               boxprops=dict(facecolor="steelblue", alpha=0.7))
    ax.axhline(0, color="red", ls="--", lw=1)
    ax.set_ylabel("Weekly Return")
    ax.set_title("RF Weekly Return Distribution")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(d / "return_stability.png", dpi=150)
    plt.close(fig)


def _write_text_report(summary: pd.DataFrame, bt: dict, top_n: int, d: Path) -> None:
    """Write a plain-text summary report."""
    lines = [
        "=" * 60,
        "  BACKTEST REPORT",
        "=" * 60,
        "",
        "--- Walk-Forward Metrics ---",
    ]
    if not summary.empty:
        lines.append(summary.to_string(index=False))
        lines.append("")
        lines += [
            "Mean metrics:",
            f"  Accuracy:  {summary['accuracy'].mean():.4f}",
            f"  Precision: {summary['precision'].mean():.4f}",
            f"  Recall:    {summary['recall'].mean():.4f}",
            f"  F1:        {summary['f1'].mean():.4f}",
            f"  AUC:       {summary['auc'].mean():.4f}",
        ]

    lines += [
        "",
        "--- Backtest Results ---",
        f"  RF Top-{top_n} cumulative:   {bt['rf_cum'].iloc[-1]:.3f}x",
        f"  Random-{top_n} cumulative:   {bt['rand_cum'].iloc[-1]:.3f}x",
        f"  Market Avg cumulative:   {bt['avg_cum'].iloc[-1]:.3f}x",
        f"  Sharpe Ratio:            {bt['sharpe']:.2f}",
        f"  Max Drawdown:            {bt['max_drawdown']:.1%}",
        "",
        "--- Profit Distribution by Window ---",
    ]
    for window_key, wstats in bt["profit_dist"]["per_window"].items():
        lines.append(
            f"  {window_key}: mean={wstats['mean_return']:.2%}, "
            f"median={wstats['median_return']:.2%}, "
            f"pos%={wstats['positive_pct']:.1%}, "
            f"n={wstats['n_predictions']}"
        )

    lines += [
        "",
        "--- Most Frequently Picked Stocks ---",
    ]
    for code, freq in list(bt["profit_dist"]["top_stocks"].items())[:10]:
        lines.append(f"  {code}: {freq} times")

    lines += ["", "=" * 60]

    report_path = d / "report.txt"
    with open(report_path, "w") as fp:
        fp.write("\n".join(lines))
    logger.info(f"Text report saved to {report_path}")
