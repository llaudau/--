"""Backtest simulation: weekly portfolio rebalancing with top-N stock picks."""
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def backtest_return(
    test_preds: pd.DataFrame,
    top_n: int = 20,
) -> dict:
    """Weekly backtest: buy top-N stocks by predicted probability each week.

    Returns dict with:
      rf_cum, rand_cum, avg_cum  — cumulative return Series
      rf_weekly, rand_weekly, avg_weekly  — raw weekly return lists
      sharpe, max_drawdown  — risk metrics
    """
    tp = test_preds.copy()
    tp["date"] = pd.to_datetime(tp["date"])
    tp["week"] = tp["date"].dt.isocalendar().week.astype(int)
    tp["year"] = tp["date"].dt.year
    tp["yearweek"] = tp["year"] * 100 + tp["week"]

    weekly_groups = tp.groupby("yearweek")

    rf_returns = []
    rand_returns = []
    all_returns = []
    rf_stocks_per_week = []

    for _, group in weekly_groups:
        if len(group) < 10:
            continue

        top_picks = group.nlargest(top_n, "y_prob")
        rf_ret = top_picks["fwd_ret"].mean()
        rf_stocks_per_week.append(top_picks["code"].tolist())

        rand_pick = group.sample(n=min(top_n, len(group)), random_state=42)
        rand_ret = rand_pick["fwd_ret"].mean()
        avg_ret = group["fwd_ret"].mean()

        rf_returns.append(rf_ret)
        rand_returns.append(rand_ret)
        all_returns.append(avg_ret)

    rf_cum = (1 + pd.Series(rf_returns)).cumprod()
    rand_cum = (1 + pd.Series(rand_returns)).cumprod()
    avg_cum = (1 + pd.Series(all_returns)).cumprod()

    sharpe = compute_sharpe(rf_returns)
    max_dd = compute_max_drawdown(rf_cum)
    profit_dist = compute_profit_distribution(test_preds, top_n)

    logger.info(f"Backtest: RF top-{top_n} = {rf_cum.iloc[-1]:.3f}x, "
                f"Random = {rand_cum.iloc[-1]:.3f}x, Market = {avg_cum.iloc[-1]:.3f}x")
    logger.info(f"Sharpe = {sharpe:.2f}, Max Drawdown = {max_dd:.1%}")

    return {
        "rf_cum": rf_cum,
        "rand_cum": rand_cum,
        "avg_cum": avg_cum,
        "rf_weekly": rf_returns,
        "rand_weekly": rand_returns,
        "avg_weekly": all_returns,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "profit_dist": profit_dist,
        "rf_stocks_per_week": rf_stocks_per_week,
    }


def compute_sharpe(weekly_returns: list[float], risk_free_annual: float = 0.03) -> float:
    """Annualized Sharpe ratio from weekly returns."""
    if not weekly_returns:
        return 0.0
    r = np.array(weekly_returns)
    rf_weekly = risk_free_annual / 52
    excess = r - rf_weekly
    if excess.std() == 0:
        return 0.0
    return float(excess.mean() / excess.std() * np.sqrt(52))


def compute_max_drawdown(cum_returns: pd.Series) -> float:
    """Maximum drawdown from cumulative return series."""
    peak = cum_returns.cummax()
    drawdown = (cum_returns - peak) / peak
    return float(drawdown.min())


def compute_profit_distribution(test_preds: pd.DataFrame, top_n: int) -> dict:
    """Analyze profit distribution across time windows and stocks."""
    tp = test_preds.copy()
    tp["date"] = pd.to_datetime(tp["date"])
    tp["year"] = tp["date"].dt.year

    # Per-window return stats
    per_window = {}
    group_col = "window" if "window" in tp.columns else "year"
    for w, wdf in tp.groupby(group_col):
        top = wdf.nlargest(min(top_n * 52, len(wdf)), "y_prob")
        per_window[f"window_{w}"] = {
            "mean_return": float(top["fwd_ret"].mean()),
            "median_return": float(top["fwd_ret"].median()),
            "positive_pct": float((top["fwd_ret"] > 0).mean()),
            "n_predictions": len(top),
        }

    # Most frequently picked stocks
    n_rows = min(top_n * 52, len(tp))
    all_top = tp.nlargest(n_rows, "y_prob")
    stock_freq = all_top["code"].value_counts().head(20)

    return {
        "per_window": per_window,
        "top_stocks": stock_freq.to_dict(),
    }
