"""MetaApi (metaapi.cloud) broker candle feed.

Pulls historical candles straight from the broker terminal that the
mt4-executor engine trades on, so scanner/journal/resolver prices are the
same prices the fills happen at. Uses MetaApi's REST market-data API over
plain httpx (no async SDK) so it drops into the synchronous DataFeed
interface without event-loop juggling.

Env:
    METAAPI_TOKEN        required
    METAAPI_ACCOUNT_ID   optional; else resolved from METAAPI_LOGIN / MT_LOGIN
    METAAPI_LOGIN        optional broker login used to look the account up
    METAAPI_SYMBOL_SUFFIX optional broker suffix appended to every pair
                          (e.g. ".pro", "m"); default "" (bare symbols)
    METAAPI_DOMAIN       optional, default agiliumtrade.agiliumtrade.ai
"""
from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timedelta, timezone

import httpx
import pandas as pd

from .types import Pair, TF

log = logging.getLogger(__name__)

DEFAULT_DOMAIN = "agiliumtrade.agiliumtrade.ai"
MAX_LIMIT = 1000  # MetaApi hard cap per candles request

# FSP timeframe -> MetaApi timeframe (MT4 allowed: 1m 5m 15m 30m 1h 4h 1d 1w 1mn)
MT_TF: dict[str, str] = {
    "M1": "1m", "M5": "5m", "M15": "15m", "M30": "30m",
    "H1": "1h", "H4": "4h", "D": "1d", "W": "1w",
}
TF_MINUTES: dict[str, int] = {
    "M1": 1, "M5": 5, "M15": 15, "M30": 30, "H1": 60, "H4": 240, "D": 1440, "W": 10080,
}

# FSP pair -> broker root symbol. Suffix is applied on top (METAAPI_SYMBOL_SUFFIX).
# DXY is intentionally absent: MetaApi FX brokers do not carry the ICE index;
# live.py already sources DXY from yfinance separately.
MT_SYMBOLS: dict[str, str] = {
    "EURUSD": "EURUSD",
    "GBPUSD": "GBPUSD",
    "USDJPY": "USDJPY",
    "USDCHF": "USDCHF",
    "AUDUSD": "AUDUSD",
    "USDCAD": "USDCAD",
    "NZDUSD": "NZDUSD",
    "EURJPY": "EURJPY",
    "GBPJPY": "GBPJPY",
}


def _fmt(dt: datetime) -> str:
    """ISO-8601 with ms + Z, the shape MetaApi's REST API expects."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def candles_to_df(candles: list[dict]) -> pd.DataFrame:
    """MetaApi candle dicts -> UTC-indexed OHLCV frame (FSP column contract)."""
    if not candles:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    rows = []
    for c in candles:
        rows.append({
            "ts": c["time"],
            "open": float(c["open"]),
            "high": float(c["high"]),
            "low": float(c["low"]),
            "close": float(c["close"]),
            # MT4 has no real volume; tickVolume is the only activity measure.
            "volume": float(c.get("tickVolume", c.get("volume", 0)) or 0),
        })
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.drop_duplicates("ts").set_index("ts").sort_index()
    df.index.name = "ts"
    return df[["open", "high", "low", "close", "volume"]]


class MetaApiFeed:
    """Broker-native bars via MetaApi REST. Paginates backwards in 1000-bar pages."""

    def __init__(
        self,
        token: str | None = None,
        account_id: str | None = None,
        login: str | None = None,
        symbol_suffix: str | None = None,
        domain: str | None = None,
        timeout: float = 20.0,
        client: httpx.Client | None = None,
    ) -> None:
        self._token = token or os.environ.get("METAAPI_TOKEN")
        if not self._token:
            raise RuntimeError("METAAPI_TOKEN not set")
        self._account_id = account_id or os.environ.get("METAAPI_ACCOUNT_ID")
        self._login = login or os.environ.get("METAAPI_LOGIN") or os.environ.get("MT_LOGIN")
        self._suffix = (symbol_suffix if symbol_suffix is not None
                        else os.environ.get("METAAPI_SYMBOL_SUFFIX", ""))
        self._domain = domain or os.environ.get("METAAPI_DOMAIN", DEFAULT_DOMAIN)
        self._client = client or httpx.Client(timeout=timeout,
                                              headers={"auth-token": self._token})
        self._region: str | None = None
        self._md_host: str | None = None
        self._host_ts = 0.0
        # Per-(pair, tf) short cache so the 4SP grader re-fetches inside a cycle
        # do not double-hit the API.
        self._cache: dict[tuple, tuple[float, pd.DataFrame]] = {}
        self._cache_ttl = 45.0

    # ----------------------------------------------------------------- symbols
    def symbol(self, pair: Pair) -> str:
        root = MT_SYMBOLS.get(pair)
        if root is None:
            raise ValueError(f"MetaApi symbol not mapped for {pair}")
        return f"{root}{self._suffix}"

    # ------------------------------------------------------------- discovery
    def _prov(self, path: str) -> dict | list:
        url = f"https://mt-provisioning-api-v1.{self._domain}{path}"
        r = self._client.get(url)
        r.raise_for_status()
        return r.json()

    def _resolve_account(self) -> None:
        """Fill account_id + region from the provisioning API (once)."""
        if self._account_id and self._region:
            return
        accounts = self._prov("/users/current/accounts")
        if self._account_id:
            match = [a for a in accounts if a.get("_id") == self._account_id]
        elif self._login:
            match = [a for a in accounts
                     if str(a.get("login")) == str(self._login)
                     and str(a.get("type", "")).startswith("cloud")]
        else:
            match = [a for a in accounts if str(a.get("type", "")).startswith("cloud")]
        if not match:
            raise RuntimeError("MetaApi account not found (set METAAPI_ACCOUNT_ID or METAAPI_LOGIN)")
        acct = match[0]
        self._account_id = acct["_id"]
        self._region = acct.get("region", "vint-hill")
        if acct.get("state") != "DEPLOYED":
            log.warning("MetaApi account %s state=%s (needs DEPLOYED for candles)",
                        self._account_id[:8], acct.get("state"))

    def _market_data_host(self) -> str:
        # MetaApi re-resolves the client-api domain every 10 min; mirror that.
        if self._md_host and time.time() - self._host_ts < 600:
            return self._md_host
        self._resolve_account()
        settings = self._prov("/users/current/servers/mt-client-api")
        self._md_host = f"https://mt-market-data-client-api-v1.{self._region}.{settings['domain']}"
        self._host_ts = time.time()
        return self._md_host

    # --------------------------------------------------------------- fetching
    def _page(self, symbol: str, tf: TF, start_time: datetime, limit: int) -> list[dict]:
        host = self._market_data_host()
        url = (f"{host}/users/current/accounts/{self._account_id}/historical-market-data/"
               f"symbols/{httpx.URL(path=symbol).path}/timeframes/{MT_TF[tf]}/candles")
        r = self._client.get(url, params={"startTime": _fmt(start_time), "limit": limit})
        r.raise_for_status()
        return r.json() or []

    def history(self, pair: Pair, tf: TF, start: datetime, end: datetime) -> pd.DataFrame:
        if tf not in MT_TF:
            raise ValueError(f"Timeframe {tf} not supported by MetaApi feed")
        sym = self.symbol(pair)
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        if end.tzinfo is None:
            end = end.replace(tzinfo=timezone.utc)

        key = (pair, tf, start.replace(second=0, microsecond=0), end.replace(second=0, microsecond=0))
        hit = self._cache.get(key)
        if hit and time.time() - hit[0] < self._cache_ttl:
            return hit[1].copy()

        # Candles load backwards from startTime; walk pages until we pass `start`.
        candles: list[dict] = []
        cursor = end
        step = timedelta(minutes=TF_MINUTES[tf])
        for _ in range(40):  # 40k bars hard ceiling per call
            page = self._page(sym, tf, cursor, MAX_LIMIT)
            if not page:
                break
            candles.extend(page)
            earliest = min(pd.Timestamp(c["time"]).tz_convert("UTC") if pd.Timestamp(c["time"]).tzinfo
                           else pd.Timestamp(c["time"]).tz_localize("UTC") for c in page)
            if earliest <= pd.Timestamp(start) or len(page) < MAX_LIMIT:
                break
            cursor = earliest.to_pydatetime() - step

        df = candles_to_df(candles)
        if not df.empty:
            df = df[(df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))]
        if tf == "D" and not df.empty:
            df = self._augment_current_day(pair, df, end)
        self._cache[key] = (time.time(), df)
        return df.copy()

    def _augment_current_day(self, pair: Pair, daily: pd.DataFrame, end: datetime) -> pd.DataFrame:
        """Append the in-progress broker day built from H1 bars.

        MT4 (as exposed by MetaApi) only returns COMPLETED daily bars, while
        intraday bars include the live one. FSP treats ``daily_df.iloc[-1]`` as
        "today" (ADR%, cycle), so without this the last row would be yesterday
        (or Sunday's stub after the weekend). Session boundary = last D stamp +
        24h, i.e. the broker's own rollover, not UTC midnight.
        """
        session_start = daily.index[-1] + pd.Timedelta(days=1)
        if pd.Timestamp(end) <= session_start:
            return daily  # last D bar already covers `end`
        try:
            h1 = self.history(pair, "H1", session_start.to_pydatetime(), end)
        except Exception as exc:  # feed error: better a stale day than no scan
            log.warning("MetaApi current-day augment failed for %s: %s", pair, exc)
            return daily
        if h1.empty:
            return daily  # market closed since rollover (weekend)
        today = pd.DataFrame([{
            "open": float(h1["open"].iloc[0]),
            "high": float(h1["high"].max()),
            "low": float(h1["low"].min()),
            "close": float(h1["close"].iloc[-1]),
            "volume": float(h1["volume"].sum()),
        }], index=pd.DatetimeIndex([session_start], name="ts"))
        return pd.concat([daily, today])

    def latest(self, pair: Pair, tf: TF, lookback_bars: int = 500) -> pd.DataFrame:
        end = datetime.now(timezone.utc)
        start = end - timedelta(minutes=TF_MINUTES[tf] * lookback_bars * 2)  # weekend buffer
        return self.history(pair, tf, start, end).tail(lookback_bars)
