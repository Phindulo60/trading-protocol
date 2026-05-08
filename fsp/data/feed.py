"""Pluggable market data feed: Dukascopy (historical) + yfinance (live-ish).

DataFeed is the single interface the rest of the engine uses. Both
implementations return UTC-indexed pandas DataFrames with columns
[open, high, low, close, volume].
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Protocol

import pandas as pd

from .types import Pair, TF

log = logging.getLogger(__name__)

CACHE_DIR = Path.home() / ".fsp" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# TradingView-style FX symbol → yfinance symbol
YF_SYMBOLS: dict[str, str] = {
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "USDJPY=X",
    "USDCHF": "USDCHF=X",
    "AUDUSD": "AUDUSD=X",
    "USDCAD": "USDCAD=X",
    "NZDUSD": "NZDUSD=X",
    "EURJPY": "EURJPY=X",
    "GBPJPY": "GBPJPY=X",
    "DXY": "DX-Y.NYB",
}

# yfinance interval strings
YF_INTERVAL = {"M1": "1m", "M5": "5m", "M15": "15m", "M30": "30m", "H1": "60m", "H4": "1h", "D": "1d"}

# dukascopy-python interval names  (library uses Enum)
DUKA_TF = {
    "M1": "m1", "M5": "m5", "M15": "m15", "M30": "m30",
    "H1": "h1", "H4": "h4", "D": "d1", "W": "w1",
}


class DataFeed(Protocol):
    def history(self, pair: Pair, tf: TF, start: datetime, end: datetime) -> pd.DataFrame: ...
    def latest(self, pair: Pair, tf: TF, lookback_bars: int = 500) -> pd.DataFrame: ...


def _cache_path(pair: str, tf: str, start: datetime, end: datetime) -> Path:
    tag = f"{pair}_{tf}_{start:%Y%m%d}_{end:%Y%m%d}.parquet"
    return CACHE_DIR / tag


class DukascopyFeed:
    """Free historical bars from Dukascopy, cached as parquet.

    Uses the `dukascopy-python` package. First fetch of a range can take a
    minute (it downloads raw tick files); subsequent loads hit the parquet cache.
    """

    def history(self, pair: Pair, tf: TF, start: datetime, end: datetime) -> pd.DataFrame:
        cache = _cache_path(pair, tf, start, end)
        if cache.exists():
            return pd.read_parquet(cache)

        import dukascopy_python as dp
        from dukascopy_python.instruments import (
            INSTRUMENT_FX_MAJORS_EUR_USD,
            INSTRUMENT_FX_MAJORS_GBP_USD,
            INSTRUMENT_FX_MAJORS_USD_JPY,
            INSTRUMENT_FX_MAJORS_USD_CHF,
            INSTRUMENT_FX_MAJORS_AUD_USD,
            INSTRUMENT_FX_MAJORS_USD_CAD,
            INSTRUMENT_FX_MAJORS_NZD_USD,
            INSTRUMENT_FX_CROSSES_EUR_JPY,
            INSTRUMENT_FX_CROSSES_GBP_JPY,
        )

        INSTR = {
            "EURUSD": INSTRUMENT_FX_MAJORS_EUR_USD,
            "GBPUSD": INSTRUMENT_FX_MAJORS_GBP_USD,
            "USDJPY": INSTRUMENT_FX_MAJORS_USD_JPY,
            "USDCHF": INSTRUMENT_FX_MAJORS_USD_CHF,
            "AUDUSD": INSTRUMENT_FX_MAJORS_AUD_USD,
            "USDCAD": INSTRUMENT_FX_MAJORS_USD_CAD,
            "NZDUSD": INSTRUMENT_FX_MAJORS_NZD_USD,
            "EURJPY": INSTRUMENT_FX_CROSSES_EUR_JPY,
            "GBPJPY": INSTRUMENT_FX_CROSSES_GBP_JPY,
        }
        if pair not in INSTR:
            raise ValueError(f"Dukascopy instrument not mapped for {pair}. "
                             f"Supported: {list(INSTR.keys())}")

        interval_map = {
            "M1": dp.INTERVAL_MIN_1, "M5": dp.INTERVAL_MIN_5, "M15": dp.INTERVAL_MIN_15,
            "M30": dp.INTERVAL_MIN_30, "H1": dp.INTERVAL_HOUR_1, "H4": dp.INTERVAL_HOUR_4,
            "D": dp.INTERVAL_DAY_1, "W": dp.INTERVAL_WEEK_1,
        }
        if tf not in interval_map:
            raise ValueError(f"Timeframe {tf} not supported by Dukascopy feed")

        log.info("Dukascopy fetch %s %s %s → %s", pair, tf, start.date(), end.date())
        df = dp.fetch(
            instrument=INSTR[pair],
            interval=interval_map[tf],
            offer_side=dp.OFFER_SIDE_BID,
            start=start,
            end=end,
        )
        # Library returns a DataFrame already. Normalise columns.
        df = df.rename(columns=str.lower)
        df = df[["open", "high", "low", "close", "volume"]].copy()
        df.index = pd.to_datetime(df.index, utc=True)
        df.index.name = "ts"
        df.to_parquet(cache)
        return df

    def latest(self, pair: Pair, tf: TF, lookback_bars: int = 500) -> pd.DataFrame:
        end = datetime.now(timezone.utc)
        per_bar = {"M1": 1, "M5": 5, "M15": 15, "M30": 30, "H1": 60, "H4": 240, "D": 1440}[tf]
        start = end - timedelta(minutes=per_bar * lookback_bars * 2)  # buffer for weekends
        return self.history(pair, tf, start, end).tail(lookback_bars)


class YFinanceFeed:
    """Near-live FX bars via yfinance. 1-min bars ~15 min delayed on free tier,
    but 5m+ are usually within a bar of real-time. No account needed."""

    def history(self, pair: Pair, tf: TF, start: datetime, end: datetime) -> pd.DataFrame:
        import yfinance as yf
        sym = YF_SYMBOLS.get(pair)
        if sym is None:
            raise ValueError(f"yfinance symbol not mapped for {pair}")
        if tf not in YF_INTERVAL:
            raise ValueError(f"Timeframe {tf} not supported by yfinance feed")
        df = yf.download(sym, start=start, end=end, interval=YF_INTERVAL[tf],
                         progress=False, auto_adjust=False)
        if df.empty:
            return df
        df.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in df.columns]
        df = df[["open", "high", "low", "close", "volume"]].copy()
        df.index = pd.to_datetime(df.index, utc=True)
        df.index.name = "ts"
        return df

    def latest(self, pair: Pair, tf: TF, lookback_bars: int = 500) -> pd.DataFrame:
        import yfinance as yf
        sym = YF_SYMBOLS[pair]
        period = {"M1": "5d", "M5": "5d", "M15": "30d", "M30": "60d", "H1": "60d", "D": "2y"}[tf]
        df = yf.download(sym, period=period, interval=YF_INTERVAL[tf],
                         progress=False, auto_adjust=False)
        if df.empty:
            return df
        df.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in df.columns]
        df = df[["open", "high", "low", "close", "volume"]].copy()
        df.index = pd.to_datetime(df.index, utc=True)
        df.index.name = "ts"
        return df.tail(lookback_bars)


def default_feed(kind: str = "duka", **kwargs) -> DataFeed:
    if kind == "duka":
        return DukascopyFeed()
    if kind == "yf":
        return YFinanceFeed()
    if kind == "td":
        from .twelve import TwelveDataFeed
        api_key = kwargs.get("api_key") or _load_td_key()
        return TwelveDataFeed(api_key)
    raise ValueError(f"Unknown feed kind: {kind!r}. Use: duka, yf, td")


def _load_td_key() -> str:
    """Load Twelve Data API key from ~/.fsp/config.toml."""
    import tomllib
    cfg_path = Path.home() / ".fsp" / "config.toml"
    if not cfg_path.exists():
        raise RuntimeError("No [twelve_data] api_key in ~/.fsp/config.toml")
    with open(cfg_path, "rb") as f:
        cfg = tomllib.load(f)
    key = cfg.get("twelve_data", {}).get("api_key")
    if not key:
        raise RuntimeError("No [twelve_data] api_key in ~/.fsp/config.toml")
    return key
