"""Twelve Data feed — real-time forex via REST API.

Free tier: 8 API calls/min, 800/day.
Batch requests: up to 8 symbols per call (counts as 1 credit each but 1 HTTP call).
Docs: https://twelvedata.com/docs#time-series
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

from .types import Pair, TF

log = logging.getLogger(__name__)

BASE_URL = "https://api.twelvedata.com"

# Twelve Data uses "EUR/USD" format
TD_SYMBOLS: dict[str, str] = {
    "EURUSD": "EUR/USD",
    "GBPUSD": "GBP/USD",
    "USDJPY": "USD/JPY",
    "USDCHF": "USD/CHF",
    "AUDUSD": "AUD/USD",
    "USDCAD": "USD/CAD",
    "NZDUSD": "NZD/USD",
    "EURJPY": "EUR/JPY",
    "GBPJPY": "GBP/JPY",
    "DXY": "DXY",
}

# Reverse lookup
TD_SYMBOLS_REV: dict[str, str] = {v: k for k, v in TD_SYMBOLS.items()}

# FSP timeframe → Twelve Data interval string
TD_INTERVAL: dict[str, str] = {
    "M1": "1min",
    "M5": "5min",
    "M15": "15min",
    "M30": "30min",
    "H1": "1h",
    "H4": "4h",
    "D": "1day",
    "W": "1week",
}


def _parse_values(values: list[dict]) -> pd.DataFrame:
    """Convert Twelve Data values list to standard OHLCV DataFrame."""
    if not values:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    df = pd.DataFrame(values)
    df["ts"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.set_index("ts").sort_index()
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype(float)
    df["volume"] = 0.0
    return df[["open", "high", "low", "close", "volume"]]


class TwelveDataFeed:
    """Real-time forex feed via Twelve Data REST API.

    Supports both single and batch (multi-symbol) requests.
    Includes a response cache (TTL=60s) to avoid duplicate fetches within a scan cycle.
    """

    def __init__(self, api_key: str):
        self._key = api_key
        self._session = requests.Session()
        self._last_call = 0.0
        self._cache: dict[str, tuple[float, pd.DataFrame]] = {}  # key -> (timestamp, df)
        self._cache_ttl = 55.0  # seconds — just under 1 minute

    def _throttle(self):
        """Respect 8 calls/min → min 8s between calls (with safety margin)."""
        elapsed = time.time() - self._last_call
        if elapsed < 8.0:
            time.sleep(8.0 - elapsed)
        self._last_call = time.time()

    def _request(self, params: dict) -> dict:
        """Make throttled API request."""
        self._throttle()
        resp = self._session.get(f"{BASE_URL}/time_series", params=params)
        resp.raise_for_status()
        return resp.json()

    def _cache_key(self, pair: str, tf: str, outputsize: int) -> str:
        return f"{pair}|{tf}|{outputsize}"

    def _get_cached(self, key: str) -> pd.DataFrame | None:
        if key in self._cache:
            ts, df = self._cache[key]
            if time.time() - ts < self._cache_ttl:
                log.debug("Cache hit: %s", key)
                return df
            del self._cache[key]
        return None

    def _set_cached(self, key: str, df: pd.DataFrame) -> None:
        self._cache[key] = (time.time(), df)

    def _fetch_single(self, pair: str, tf: str, outputsize: int = 500,
                      start_date: str | None = None, end_date: str | None = None) -> pd.DataFrame:
        sym = TD_SYMBOLS.get(pair)
        if sym is None:
            raise ValueError(f"Twelve Data: no mapping for {pair}")
        interval = TD_INTERVAL.get(tf)
        if interval is None:
            raise ValueError(f"Twelve Data: unsupported timeframe {tf}")

        # Check cache (avoids duplicate fetches within same scan cycle)
        cache_key = self._cache_key(pair, tf, outputsize)
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        params: dict = {
            "symbol": sym,
            "interval": interval,
            "outputsize": outputsize,
            "apikey": self._key,
            "timezone": "UTC",
        }
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date

        data = self._request(params)

        if data.get("status") != "ok":
            raise RuntimeError(f"Twelve Data error: {data.get('message', data)}")

        df = _parse_values(data.get("values", []))
        self._set_cached(cache_key, df)
        return df

    def history(self, pair: Pair, tf: TF, start: datetime, end: datetime) -> pd.DataFrame:
        return self._fetch_single(
            pair, tf, outputsize=5000,
            start_date=start.strftime("%Y-%m-%d %H:%M:%S"),
            end_date=end.strftime("%Y-%m-%d %H:%M:%S"),
        )

    def latest(self, pair: Pair, tf: TF, lookback_bars: int = 500) -> pd.DataFrame:
        return self._fetch_single(pair, tf, outputsize=min(lookback_bars, 5000))

    def batch_latest(self, pairs: list[str], tf: str,
                     lookback_bars: int = 500) -> dict[str, pd.DataFrame]:
        """Fetch multiple symbols in one API call. Returns {pair: DataFrame}.

        Batch counts as N credits but only 1 HTTP request (1 rate-limit slot).
        Max 8 symbols per batch on free tier.
        """
        symbols = []
        for p in pairs:
            sym = TD_SYMBOLS.get(p)
            if sym:
                symbols.append(sym)
            else:
                log.warning("Twelve Data: skipping unmapped pair %s", p)

        interval = TD_INTERVAL.get(tf)
        if interval is None:
            raise ValueError(f"Twelve Data: unsupported timeframe {tf}")

        params = {
            "symbol": ",".join(symbols),
            "interval": interval,
            "outputsize": min(lookback_bars, 5000),
            "apikey": self._key,
            "timezone": "UTC",
        }

        data = self._request(params)

        result: dict[str, pd.DataFrame] = {}

        # Single symbol → response is flat; multiple → keyed by symbol
        if len(symbols) == 1:
            sym = symbols[0]
            pair = TD_SYMBOLS_REV.get(sym, pairs[0])
            if data.get("status") == "ok":
                result[pair] = _parse_values(data.get("values", []))
            else:
                log.error("Twelve Data error for %s: %s", pair, data.get("message"))
        else:
            for sym in symbols:
                pair = TD_SYMBOLS_REV.get(sym, sym)
                sym_data = data.get(sym, {})
                if sym_data.get("status") == "ok":
                    result[pair] = _parse_values(sym_data.get("values", []))
                else:
                    log.error("Twelve Data error for %s: %s",
                              pair, sym_data.get("message", "no data"))

        return result
