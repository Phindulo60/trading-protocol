"""Tests for Twelve Data feed adapter (unit tests with mocked HTTP)."""
import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone

from fsp.data.twelve import TwelveDataFeed, _parse_values, TD_SYMBOLS


SAMPLE_RESPONSE = {
    "meta": {"symbol": "EUR/USD", "interval": "15min"},
    "values": [
        {"datetime": "2026-05-08 12:00:00", "open": "1.1760", "high": "1.1770", "low": "1.1755", "close": "1.1765"},
        {"datetime": "2026-05-08 11:45:00", "open": "1.1750", "high": "1.1762", "low": "1.1748", "close": "1.1760"},
    ],
    "status": "ok",
}

BATCH_RESPONSE = {
    "EUR/USD": {
        "meta": {"symbol": "EUR/USD"},
        "values": [{"datetime": "2026-05-08 12:00:00", "open": "1.1760", "high": "1.1770", "low": "1.1755", "close": "1.1765"}],
        "status": "ok",
    },
    "GBP/USD": {
        "meta": {"symbol": "GBP/USD"},
        "values": [{"datetime": "2026-05-08 12:00:00", "open": "1.3600", "high": "1.3620", "low": "1.3590", "close": "1.3610"}],
        "status": "ok",
    },
}


def test_parse_values():
    df = _parse_values(SAMPLE_RESPONSE["values"])
    assert len(df) == 2
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert df.index.name == "ts"
    assert df["close"].iloc[-1] == 1.1765


def test_parse_values_empty():
    df = _parse_values([])
    assert df.empty


def test_td_symbols_coverage():
    """All our active pairs are mapped."""
    for pair in ["EURUSD", "GBPUSD", "AUDUSD", "USDCAD", "EURJPY", "GBPJPY"]:
        assert pair in TD_SYMBOLS


@patch("fsp.data.twelve.TwelveDataFeed._request")
def test_latest(mock_req):
    mock_req.return_value = SAMPLE_RESPONSE
    feed = TwelveDataFeed("fake_key")
    feed._last_call = 999999999999  # skip throttle
    df = feed.latest("EURUSD", "M15", lookback_bars=10)
    assert len(df) == 2
    assert mock_req.called


@patch("fsp.data.twelve.TwelveDataFeed._request")
def test_batch_latest(mock_req):
    mock_req.return_value = BATCH_RESPONSE
    feed = TwelveDataFeed("fake_key")
    feed._last_call = 999999999999
    result = feed.batch_latest(["EURUSD", "GBPUSD"], "M15", lookback_bars=5)
    assert "EURUSD" in result
    assert "GBPUSD" in result
    assert result["EURUSD"]["close"].iloc[-1] == 1.1765


@patch("fsp.data.twelve.TwelveDataFeed._request")
def test_error_handling(mock_req):
    mock_req.return_value = {"status": "error", "message": "API limit reached"}
    feed = TwelveDataFeed("fake_key")
    feed._last_call = 999999999999
    with pytest.raises(RuntimeError, match="API limit"):
        feed.latest("EURUSD", "M15")


def test_invalid_pair():
    feed = TwelveDataFeed("fake_key")
    feed._last_call = 999999999999
    with pytest.raises(ValueError, match="no mapping"):
        feed._fetch_single("INVALID", "M15")


def test_invalid_tf():
    feed = TwelveDataFeed("fake_key")
    feed._last_call = 999999999999
    with pytest.raises(ValueError, match="unsupported timeframe"):
        feed._fetch_single("EURUSD", "M3")
