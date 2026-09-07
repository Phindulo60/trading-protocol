"""MetaApiFeed: pagination, normalisation, symbol suffix, current-day augment.

All HTTP is served by an httpx MockTransport; no network, no SDK.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import httpx
import pandas as pd
import pytest

from fsp.data.metaapi import MetaApiFeed, candles_to_df, MT_SYMBOLS
from fsp.data import feed as feedmod

ACCT = "acct-1234567890"


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _candles(end: datetime, n: int, minutes: int, price: float = 1.1) -> list[dict]:
    """n candles ending at `end` (inclusive), ascending, `minutes` apart."""
    out = []
    for i in range(n):
        t = end - timedelta(minutes=minutes * (n - 1 - i))
        out.append({"symbol": "EURUSD", "timeframe": "1h", "time": _iso(t),
                    "brokerTime": "x", "open": price, "high": price + 0.001,
                    "low": price - 0.001, "close": price, "tickVolume": 10, "spread": 1})
    return out


class FakeMetaApi:
    """Serves provisioning + market-data endpoints; records candle requests."""

    def __init__(self, tf_minutes: int = 60, total_bars: int = 2500, symbols=("EURUSD",),
                 daily: list[dict] | None = None, h1_today: list[dict] | None = None):
        self.tf_minutes = tf_minutes
        self.total_bars = total_bars
        self.symbols = set(symbols)
        self.daily = daily
        self.h1_today = h1_today
        self.requests: list[httpx.Request] = []
        self.transport = httpx.MockTransport(self.handle)

    def handle(self, req: httpx.Request) -> httpx.Response:
        self.requests.append(req)
        assert req.headers.get("auth-token") == "tok"
        p = req.url.path
        if p == "/users/current/accounts":
            return httpx.Response(200, json=[
                {"_id": "other", "login": "1", "type": "cloud-g2", "region": "new-york", "state": "DEPLOYED"},
                {"_id": ACCT, "login": "760459", "type": "cloud-g2", "region": "london", "state": "DEPLOYED"},
            ])
        if p == "/users/current/servers/mt-client-api":
            return httpx.Response(200, json={"domain": "agiliumtrade.ai", "hostname": "mt-client-api-v1"})
        if "/historical-market-data/symbols/" in p:
            assert req.url.host == f"mt-market-data-client-api-v1.london.agiliumtrade.ai"
            sym = p.split("/symbols/")[1].split("/")[0]
            tf = p.split("/timeframes/")[1].split("/")[0]
            if sym not in self.symbols:
                return httpx.Response(500, json={"error": "unexpectedError",
                                                 "message": f"Symbol {sym} does not exist"})
            limit = int(req.url.params.get("limit", 1000))
            start = req.url.params.get("startTime")
            if tf == "1d" and self.daily is not None:
                return httpx.Response(200, json=self.daily)
            if tf == "1h" and self.h1_today is not None:
                return httpx.Response(200, json=self.h1_today)
            end = pd.Timestamp(start).to_pydatetime() if start else datetime(2026, 9, 7, 20, tzinfo=timezone.utc)
            # floor to grid, serve min(limit, remaining) bars ending at/before `end`
            end = end - timedelta(minutes=end.minute % self.tf_minutes, seconds=end.second,
                                  microseconds=end.microsecond)
            first_allowed = datetime(2026, 9, 7, 20, tzinfo=timezone.utc) - timedelta(
                minutes=self.tf_minutes * (self.total_bars - 1))
            n_avail = int((end - first_allowed).total_seconds() // 60 // self.tf_minutes) + 1
            n = max(0, min(limit, n_avail))
            return httpx.Response(200, json=_candles(end, n, self.tf_minutes))
        return httpx.Response(404)


def _feed(fake: FakeMetaApi, **kw) -> MetaApiFeed:
    client = httpx.Client(transport=fake.transport, headers={"auth-token": "tok"})
    return MetaApiFeed(token="tok", login="760459", symbol_suffix=kw.pop("symbol_suffix", ""),
                       client=client, **kw)


def test_candles_to_df_contract():
    end = datetime(2026, 9, 7, 20, tzinfo=timezone.utc)
    df = candles_to_df(_candles(end, 3, 60) + _candles(end, 1, 60))  # dup last bar
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert df.index.name == "ts" and str(df.index.tz) == "UTC"
    assert len(df) == 3 and df.index.is_monotonic_increasing
    assert df["volume"].iloc[-1] == 10.0
    assert candles_to_df([]).empty


def test_account_resolved_by_login_and_region_host_used():
    fake = FakeMetaApi()
    f = _feed(fake)
    end = datetime(2026, 9, 7, 20, tzinfo=timezone.utc)
    df = f.history("EURUSD", "H1", end - timedelta(hours=5), end)
    assert f._account_id == ACCT and f._region == "london"
    assert len(df) == 6
    paths = [r.url.path for r in fake.requests]
    assert paths[0] == "/users/current/accounts"
    assert paths[1] == "/users/current/servers/mt-client-api"
    assert f"/accounts/{ACCT}/historical-market-data/symbols/EURUSD/timeframes/1h/candles" in paths[2]


def test_paginates_backwards_past_1000_bars():
    fake = FakeMetaApi(tf_minutes=60, total_bars=2500)
    f = _feed(fake)
    end = datetime(2026, 9, 7, 20, tzinfo=timezone.utc)
    start = end - timedelta(hours=1500)
    df = f.history("EURUSD", "H1", start, end)
    candle_reqs = [r for r in fake.requests if "candles" in r.url.path]
    assert len(candle_reqs) == 2  # 1000 + 501
    assert len(df) == 1501
    assert df.index[0] == pd.Timestamp(start) and df.index[-1] == pd.Timestamp(end)
    assert not df.index.duplicated().any()


def test_stops_when_history_exhausted():
    fake = FakeMetaApi(tf_minutes=60, total_bars=300)
    f = _feed(fake)
    end = datetime(2026, 9, 7, 20, tzinfo=timezone.utc)
    df = f.history("EURUSD", "H1", end - timedelta(hours=5000), end)
    assert len(df) == 300
    assert len([r for r in fake.requests if "candles" in r.url.path]) == 1


def test_symbol_suffix_applied_and_unknown_symbol_raises():
    fake = FakeMetaApi(symbols=("EURUSD.pro",))
    end = datetime(2026, 9, 7, 20, tzinfo=timezone.utc)
    f = _feed(fake, symbol_suffix=".pro")
    assert f.symbol("EURUSD") == "EURUSD.pro"
    assert len(f.history("EURUSD", "H1", end - timedelta(hours=2), end)) == 3
    bare = _feed(FakeMetaApi(symbols=("EURUSD.pro",)))
    with pytest.raises(httpx.HTTPStatusError):
        bare.history("EURUSD", "H1", end - timedelta(hours=2), end)
    with pytest.raises(ValueError):
        bare.symbol("DXY")
    assert set(MT_SYMBOLS) >= {"EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD",
                               "NZDUSD", "EURJPY", "GBPJPY"}


def test_daily_augments_in_progress_broker_day():
    d = datetime(2026, 9, 5, 23, tzinfo=timezone.utc)  # Sunday stub bar (broker day = 23:00 UTC)
    daily = [
        {"time": _iso(d - timedelta(days=2)), "open": 1.1, "high": 1.2, "low": 1.0, "close": 1.15, "tickVolume": 5},
        {"time": _iso(d), "open": 1.15, "high": 1.16, "low": 1.14, "close": 1.155, "tickVolume": 6},
    ]
    h1_today = [
        {"time": _iso(d + timedelta(days=1, hours=1)), "open": 1.156, "high": 1.17, "low": 1.155, "close": 1.16, "tickVolume": 1},
        {"time": _iso(d + timedelta(days=1, hours=2)), "open": 1.16, "high": 1.165, "low": 1.13, "close": 1.14, "tickVolume": 2},
    ]
    fake = FakeMetaApi(daily=daily, h1_today=h1_today)
    f = _feed(fake)
    end = d + timedelta(days=1, hours=3)
    df = f.history("EURUSD", "D", end - timedelta(days=10), end)
    assert len(df) == 3
    today = df.iloc[-1]
    assert df.index[-1] == pd.Timestamp(d + timedelta(days=1))
    assert (today.open, today.high, today.low, today.close, today.volume) == (1.156, 1.17, 1.13, 1.14, 3.0)


def test_daily_not_augmented_when_market_closed_since_rollover():
    d = datetime(2026, 9, 5, 23, tzinfo=timezone.utc)
    daily = [{"time": _iso(d), "open": 1, "high": 1, "low": 1, "close": 1, "tickVolume": 1}]
    fake = FakeMetaApi(daily=daily, h1_today=[])
    f = _feed(fake)
    df = f.history("EURUSD", "D", d - timedelta(days=5), d + timedelta(days=1, hours=2))
    assert len(df) == 1


def test_cache_within_ttl_avoids_refetch():
    fake = FakeMetaApi()
    f = _feed(fake)
    end = datetime(2026, 9, 7, 20, tzinfo=timezone.utc)
    f.history("EURUSD", "H1", end - timedelta(hours=5), end)
    n = len(fake.requests)
    f.history("EURUSD", "H1", end - timedelta(hours=5), end)
    assert len(fake.requests) == n


def test_default_feed_mt_singleton(monkeypatch):
    monkeypatch.setenv("METAAPI_TOKEN", "tok")
    feedmod._feed_cache.pop("mt", None)
    a = feedmod.default_feed("mt")
    b = feedmod.default_feed("mt")
    assert a is b and isinstance(a, MetaApiFeed)
    feedmod._feed_cache.pop("mt", None)


def test_missing_token_raises(monkeypatch):
    monkeypatch.delenv("METAAPI_TOKEN", raising=False)
    with pytest.raises(RuntimeError):
        MetaApiFeed()
