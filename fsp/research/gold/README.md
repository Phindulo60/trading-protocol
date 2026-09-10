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

## Addendum 2026-09-09: literature replication + the "$50/week on $500" question

Target = +10%/week = 14,000%/yr compounded. To *expect* that at Sharpe 1.1 you must run 66% weekly
vol, i.e. an 18% chance per week of losing half the account. At Sharpe 3 it is still 24% weekly vol.

**arXiv 2511.08571 "Forecast-to-Fill"** (Sharpe 2.88 on gold, EMA-slope + 50d momentum regime,
15% vol target, 0.4 Kelly, ATR stops). Replicated in `replicate_f2f.py` on Dukascopy daily, rolling
10y/6m walk-forward, both the paper's 0.7 bps and our observed 2.4 bps cost, lambda swept 0.90/0.94/0.97:
Sharpe **0.6–1.05**, beta to gold **0.5–0.7** (paper: 0.03), hit 53% (paper: 65.8%), 2025 alone is most
of the return. Does not replicate; it is buy-and-hold gold with a stop. Not peer-reviewed; discard.

**Baltussen, Da, Lammers, Martens (JFE 2021) intraday momentum** (day-to-date return predicts the last
30–60 min before COMEX settle). `intraday_mom.py` on M15 2022+ and H1 2017+: rho 0.00–0.03, gross
≈ 0, **net −1.4 to −1.9 bps/day, negative every year 2017–2026**. Dead in gold at retail cost.

**Leverage on the one thing that works** (`leverage_sim.py`, block-bootstrap of real OOS
HAR-vol-target daily returns, $500 account, TradeNation 0.01 lot = 1 oz = $4,411 notional):

| position | leverage | P(week ≥ +$50) | median week | P(ruin ≤ 12 wk) | P(ruin ≤ 52 wk) | median equity @ 52 wk |
|---|---|---|---|---|---|---|
| 0.01 lot (minimum) | 8.8x | 33% | +$5 | **22%** | **50%** | $790 |
| 0.02 lot (max margin) | 17.6x | 36% | +$1 | 60% | 89% | $15 |

On $500, the *smallest* gold position the broker allows already forces ~9x leverage; a coin-flip of
ruin inside a year is structural, not a modelling choice. The strategy is sound; the account is too
small for the instrument.

## Addendum 2026-09-10: ICT concepts on gold — the first real edge

Ported the ICT confluence engine (`fsp/ict/`) to XAUUSD. The FX-tuned version lost -0.5R/trade
because its stops ($2/oz) sit inside gold's $4-8 bar noise. Fix = ATR-buffered stops (0.5 ATR
beyond the swept level) + a min-risk filter (>=0.5 ATR) so the $0.30 spread stays <10% of risk.
`ict_run.py` (gold-correct pip=0.01 via `data.types.pip_size`), Dukascopy mid, spread 30pt.

**Backtests (walk-forward, limit fills, SL slippage, one position at a time):**
- H1/H4 2017-2026, 216 trades: ALL +0.125R (t=1.0). Grade is MONOTONIC and predictive:
  B -0.45R / A +0.31R / A+ +1.24R. **A/A+ only = +0.449R, t=2.67, DD -10R**, stable OOS
  (2017-22 +0.185R, 2023-26 +0.772R). 3R TP cap HURTS (+0.03 vs +0.13) -> let winners run.
- M15/H1 2022-2026, 473 trades: +0.205R (t=2.23). Grade non-monotonic (noise); the robust cut
  is **session**: NY (AM+PM) = +0.467R, t=2.99, n=212, positive every sub-period.
- **COMBINED stream** (H1 A/A+ all-session + M15 NY A/A+): n=286, **+0.363R, t=3.19**, ~30 trades/yr,
  only 2020 negative, late-sample stronger than early. This is the deployable edge.

**Sizing = `sizing.py`**: fixed-fraction of CURRENT equity, snapped to 0.01 lot grid, capped 0.10,
margin-checked, risk halved after 15% drawdown. Block-bootstrap MC, $1000 account, combined stream:

| risk | 1yr median | 3yr median | P(profit 3yr) | P(2x 3yr) | P(blow-up) | worst DD |
|---|---|---|---|---|---|---|
| 1% | $1,089 | $1,323 (+32%) | 96% | 1% | 0% | -8% |
| 2% | $1,177 | $1,608 (+61%) | 95% | 19% | 0% | -13% |
| 3% | $1,207 | $1,711 (+71%) | 84%/96% | 30% | 0% | -17% |

**Verdict:** ICT on gold is a genuine, robust, non-blowup growth engine (~15-70%/yr by risk
appetite). It is NOT a weekly-income engine: ~30 A/A+ setups a year at +0.36R is ~$2-5/week on
$1000. Weekly income needs frequency this doesn't have or leverage that ruins the account.
Deploy for compounding, not for a weekly wage.
