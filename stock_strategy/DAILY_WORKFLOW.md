# Daily Trading Workflow

## Strategy Summary
- **Model:** Random Forest (RF)
- **Parameters:** forward_days=3, threshold=2%, top_n=10
- **Backtest Performance:** 126x cumulative return, Sharpe 4.10, Max DD -14.8%
- **Hold Period:** 3 trading days
- **Position Size:** 10% per stock (equal weight)

## Daily Routine

### Every Trading Day After Market Close (3:00 PM China time)

```bash
# 1. Download latest data
python -m data_downloader.download_all

# 2. Clean the new data
python -m data_washer.run

# 3. Rebuild stock pool (filters ST stocks, illiquid, etc.)
python -m stock_pool.run

# 4. Generate tomorrow's prediction
python -m model.RF.generate_daily_prediction
```

This creates a file: `data/predictions/prediction_YYYYMMDD.csv`

### Next Morning Before Market Open (9:30 AM)

```bash
# View today's buy/sell instructions
python -m model.RF.view_predictions --latest
```

**Portfolio Action:**
1. **SELL** the 10 stocks you bought 3 trading days ago (at market open or close)
2. **BUY** the 10 new stocks from today's prediction (at market open)
3. Each stock gets 10% of your capital (equal weight)

### Example Timeline

| Date | Action |
|------|--------|
| **Mon Apr 14** | BUY: 002542, 603008, 002207, 000639, 000048, 002364, 600703, 600408, 000016, 000980 |
| Tue Apr 15 | Hold (day 1) |
| Wed Apr 16 | Hold (day 2) |
| **Thu Apr 17** | SELL the Apr 14 batch + BUY new 10 stocks from Apr 16 prediction |

## View Prediction History

```bash
# Show all predictions
python -m model.RF.view_predictions

# Show specific date
python -m model.RF.view_predictions --date 20260413

# Show latest only
python -m model.RF.view_predictions --latest
```

## Prediction File Format

Each CSV contains:
- `code`: Stock code (6 digits)
- `pred_prob`: Model confidence (0-1)
- `signal_date`: Date the prediction was made
- `buy_date`: When to buy (next trading day)
- `sell_date`: When to sell (3 trading days after buy)
- `forward_days`: Hold period (3)
- `return_threshold`: Target return (2%)
- `position_size_pct`: Position size (10%)

## Important Notes

⚠️ **This is a mean-reversion strategy** — it buys stocks that have dropped 20-30% recently, expecting a 2%+ bounce within 3 days.

⚠️ **Backtest ≠ Live Performance** — The 126x return is from backtesting. Real trading has:
- Transaction costs (~0.1-0.3% per trade)
- Slippage (price moves between signal and execution)
- Market impact (your orders affect prices)

⚠️ **Risk Management:**
- Start with small capital to validate live performance
- The strategy has 74% weekly win rate but can have -15% drawdowns
- Worst historical week: -14.8% (Jan 2024)
- Don't use money you can't afford to lose

⚠️ **Falling Knives:** Some picked stocks may keep falling (fraud, delisting risk). The model doesn't know fundamentals — it only sees price patterns.

## Automation (Optional)

To run daily predictions automatically:

```bash
# Add to crontab (runs at 3:30 PM China time daily)
30 15 * * 1-5 cd /home/khw/Documents/Git_repository/stock_strategy && \
  source data_downloader/.venv/bin/activate && \
  python -m data_downloader.download_all && \
  python -m data_washer.run && \
  python -m stock_pool.run && \
  python -m model.RF.generate_daily_prediction
```

Then check `data/predictions/` folder each morning for the latest file.
