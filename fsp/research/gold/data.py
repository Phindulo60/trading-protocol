"""Gold (XAUUSD) research data layer.

Sources:
  - Dukascopy bid+ask bars (deep history) cached under ~/.fsp/cache/gold/
  - yfinance daily macro context (DXY, US10Y yield, VIX, silver, S&P)

Everything is UTC-indexed. Spread is derived from bid/ask closes so cost
modelling uses the observed spread, not a guess.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

GOLD_CACHE = Path.home() / ".fsp" / "cache" / "gold"

MACRO_TICKERS = {
    "dxy": "DX-Y.NYB",   # US dollar index
    "us10y": "^TNX",     # 10y nominal yield (x10 in yfinance units)
    "vix": "^VIX",
    "silver": "SI=F",
    "spx": "^GSPC",
    "tip": "TIP",        # TIPS ETF: real-yield proxy (inverse)
}


def load_bars(tf: str, side: str = "bid") -> pd.DataFrame:
    """OHLCV for XAUUSD at `tf` from the Dukascopy cache."""
    p = GOLD_CACHE / f"XAUUSD_{tf}_{side}.parquet"
    if not p.exists():
        raise FileNotFoundError(f"{p} missing; run the gold fetch script first")
    df = pd.read_parquet(p)
    df.index = pd.to_datetime(df.index, utc=True)
    return df[df["volume"] > 0]  # Dukascopy pads closed hours with zero-volume rows


def load_mid_with_spread(tf: str) -> pd.DataFrame:
    """Mid-price OHLC plus observed spread (ask.close - bid.close) in $."""
    bid, ask = load_bars(tf, "bid"), load_bars(tf, "ask")
    idx = bid.index.intersection(ask.index)
    bid, ask = bid.loc[idx], ask.loc[idx]
    mid = pd.DataFrame({
        "open": (bid["open"] + ask["open"]) / 2,
        "high": (bid["high"] + ask["high"]) / 2,
        "low": (bid["low"] + ask["low"]) / 2,
        "close": (bid["close"] + ask["close"]) / 2,
        "volume": bid["volume"],
        "spread": (ask["close"] - bid["close"]).clip(lower=0),
    })
    mid.index.name = "ts"
    return mid


def load_macro(start: str = "2009-01-01", cache: bool = True) -> pd.DataFrame:
    """Daily macro closes, forward-filled onto a UTC daily index.

    Stored once so re-runs are offline. Values are as-of the NY close of that
    date, so when joining to intraday bars they must be lagged by one day to
    avoid look-ahead (see features.attach_macro).
    """
    p = GOLD_CACHE / "macro_daily.parquet"
    if cache and p.exists():
        return pd.read_parquet(p)
    import yfinance as yf
    frames = {}
    for name, tkr in MACRO_TICKERS.items():
        raw = yf.download(tkr, start=start, progress=False, auto_adjust=False)
        if raw.empty:
            log.warning("macro %s (%s) empty", name, tkr)
            continue
        close = raw["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        frames[name] = close
    macro = pd.DataFrame(frames)
    macro.index = pd.to_datetime(macro.index).tz_localize("UTC") if macro.index.tz is None \
        else macro.index.tz_convert("UTC")
    macro.index = macro.index.normalize()
    macro = macro.sort_index().ffill()
    macro.to_parquet(p)
    return macro
