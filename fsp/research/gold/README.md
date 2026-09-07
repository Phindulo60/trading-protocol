# Gold (XAUUSD) forecasting research — 2026-09-08

**Question:** can we build a better price predictor for gold than the June meta-labeller?
**Answer:** not for *direction*. Volatility is forecastable and improves a long position via sizing.

## Data
Dukascopy bid+ask, cached `~/.fsp/cache/gold/`: D 2010→, H1 2016→, M15 2022→, M5 2024-09→.
Cost = observed ask-bid at entry bar (median 1.87 bps H1, 2.28 bps D). TradeNation live spread ~0.6 bps
(0.28 on 4411), so the cost bar used here is conservative.
Macro (yfinance, lagged 1 day): DXY, US10Y, VIX, silver, SPX, TIP.

## Method
`evaluate.walk_forward`: expanding window, 6 folds, h-bar purge, thresholds fit on train only.
t-stats are Newey-West (lag h-1) because h-bar labels sampled every bar overlap; the naive
t-stat is ~sqrt(h) too big (33k "trades" at h=24 were really ~1.4k).
Key metric = **excess over always-long**, since gold went 1,150→4,400 in-sample and any model
that leans long inherits that drift.

## Results (all OOS)
| tf | h | best model | excess vs long (bps) | excess t | verdict |
|---|---|---|---|---|---|
| H1 | 1 | lgbm_clf@q50 | +0.7 | 5.6 | trades 50% of hours, but net −1.5 bps/trade: spread > edge |
| H1 | 4 | lgbm_reg@q50 | +0.4 | 0.9 | noise |
| H1 | 24 | lgbm_reg@q50 | −1.4 | −0.5 | noise; Sharpe 0.70 is drift in disguise |
| D | 1 | lgbm_clf@q50 | −2.4 | −1.1 | noise |
| D | 5 | ridge | −16 | −1.9 | worse than long |
| D | 20 | ridge@q50 | −70 | −2.4 | worse than long |

Always-long itself: t = 1.1–1.8 (not significant). Time-series momentum: negative every horizon.
Direction IC ≤ 0.03 everywhere. No model at any horizon beats holding gold net of cost.

## What IS there
1. **Asia-reopen drift (21:00–00:00 UTC)**: gross +1.4–1.8 bps/day, positive 8/10 years, and it
   is most of the day's return (rest-of-day t≈0.5). But net of the 1.9 bps spread it loses money
   (t = −2 to −3.7). At TradeNation's ~0.6 bps it would be marginally positive; too thin to trade alone.
2. **Volatility is forecastable**: HAR-RV on hourly realised vol, OOS 2021-11→2026-09:
   R² 0.14 (naive yesterday-vol: −0.74), realised vol by forecast quintile 61→128 bps/day.
3. **Vol-targeting a long gold position with the HAR forecast** (same OOS, cost-adjusted):
   Sharpe 0.85 → 1.09, maxDD −32% → −29%, t 2.06 → 2.66, mean exposure 1.1x, better in 4/6 years.
   Naive yesterday-vol sizing makes it worse (0.69), so the HAR model is the ingredient.

## Recommendation
Do not deploy a direction model on gold. If we trade gold at all, the evidence supports a
**long-biased, HAR-vol-targeted position rebalanced daily** (turnover ~0.09/day), with the
FSP strategies used only as timing overlays if they survive their own gold backtests.
Next: triple-barrier meta-labelling of FSP signals on XAUUSD M15 (`features.triple_barrier`) to
see whether *entries* have edge even though *forecasting* does not.

## Files
`data.py` loaders · `features.py` features/labels · `evaluate.py` walk-forward + NW stats + baselines ·
`run_h1.py`, `run_daily.py` model sweeps · `diagnostics.py` seasonality + HAR-RV · `session_drift.py` ·
`vol_target.py` · tests `fsp/tests/test_gold_research.py`
