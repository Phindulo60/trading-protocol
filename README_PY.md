# 4-Step Protocol — Python Engine (fsp)

Standalone FX trade-signal engine for the 4-Step Protocol. No TradingView, no
broker lock-in. Reads market data, runs the 4-step analysis, sends Telegram
alerts. You execute manually on TradeNation.

## Stack
- **Python 3.11+**, uv for deps
- **Dukascopy** (free, no account) for historical + backtest data
- **yfinance** (free, no account) for near-live data (M5+ OK)
- **Telegram bot** for alerts
- **Parquet + SQLite** for cache + trade journal

## Install (Mac)

```bash
# one-time: install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

cd ~/.aki/aki_workspace/trading-protocol
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

## First smoke test

```bash
# Fetch 90d of EURUSD H1 from Dukascopy, cache it locally
fsp fetch --pair EURUSD --tf H1 --days 90

# Scan for swings + unmitigated FVGs on that data
fsp scan --pair EURUSD --tf H1 --days 30

# Run the unit tests
pytest fsp/tests -v
```

## What's built so far (Milestone 1)

- [x] Project scaffold + uv
- [x] Pluggable `DataFeed` interface (Dukascopy + yfinance)
- [x] Parquet caching of historical bars
- [x] Swing detection (strong vs weak, with liquidity-take test)
- [x] FVG detection with mitigation + inversion tracking
- [x] CLI (`fsp fetch`, `fsp scan`)
- [x] Unit tests

## What's next

- [ ] M2: Order blocks, session tracking, PDH/PDL/PWH/PWL levels
- [ ] M3: Market cycle + H1 OF bias + DXY SMT module
- [ ] M4: Setup grader (A+/A/B checklist)
- [ ] M5: Telegram notifier + `fsp live` loop
- [ ] M6: Backtest engine + HTML report
- [ ] M7: Trade journal (SQLite) + outcome tracker

## Project layout

```
fsp/
├── data/         # types, feed (Dukascopy + yfinance)
├── structure/    # swings, FVGs, OBs
├── context/      # sessions, cycle, OF bias, SMT
├── grader/       # A+/A/B rubric
├── manager/      # entry/SL/TP ladder
├── notify/       # Telegram
├── backtest/     # event-loop backtester
├── journal/      # SQLite trade log
├── cli/          # typer app
└── tests/
```
