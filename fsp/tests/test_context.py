"""Tests for the LLM context builder."""
import pytest
from unittest.mock import patch
from datetime import datetime, timezone

from fsp.llm.context import (
    current_session, upcoming_events, recent_trades,
    strategy_stats, price_context,
)


def test_current_session():
    session = current_session()
    assert "name" in session
    assert session["name"] in ("ASIA", "LO", "NY-AM", "NY-PM", "OFF")
    assert "time" in session


@patch("fsp.llm.context.requests.get")
def test_upcoming_events_empty(mock_get):
    """No matching events returns empty list."""
    mock_get.return_value.status_code = 200
    mock_get.return_value.json.return_value = []
    events = upcoming_events("EURUSD")
    assert events == []


@patch("fsp.llm.context.requests.get")
def test_upcoming_events_filters_by_currency(mock_get):
    """Only returns events matching the pair's currencies."""
    import fsp.llm.context as ctx_mod
    ctx_mod._cached_calendar = None  # clear cache

    mock_get.return_value.status_code = 200
    mock_get.return_value.raise_for_status = lambda: None
    now = datetime.now(timezone.utc)
    from datetime import timedelta
    future = (now + timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%S+00:00")
    mock_get.return_value.json.return_value = [
        {"title": "NFP", "country": "USD", "date": future, "impact": "High"},
        {"title": "BOJ Rate", "country": "JPY", "date": future, "impact": "High"},
        {"title": "Milk Price", "country": "NZD", "date": future, "impact": "Medium"},
    ]
    events = upcoming_events("EURUSD")
    currencies = {e["currency"] for e in events}
    assert "NZD" not in currencies  # EURUSD doesn't care about NZD
    assert "USD" in currencies or "EUR" in currencies


def test_price_context_empty():
    """Empty dataframe returns empty context."""
    import pandas as pd
    df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    ctx = price_context("EURUSD", df)
    assert ctx == {}


def test_price_context_with_data():
    """Real-ish data returns trend and ATR."""
    import pandas as pd
    import numpy as np
    dates = pd.date_range("2026-01-01", periods=30, freq="15min", tz="UTC")
    prices = 1.1 + np.cumsum(np.random.randn(30) * 0.001)
    df = pd.DataFrame({
        "open": prices - 0.0005,
        "high": prices + 0.001,
        "low": prices - 0.001,
        "close": prices,
        "volume": 0,
    }, index=dates)
    df.index.name = "ts"
    ctx = price_context("EURUSD", df)
    assert "m15_trend" in ctx
    assert ctx["m15_trend"] in ("bullish", "bearish")
    assert "m15_atr" in ctx


def test_strategy_stats_empty():
    """No trades returns zero stats."""
    with patch("fsp.llm.context.recent_trades", return_value=[]):
        stats = strategy_stats("TREND_RSI")
        assert stats["total"] == 0
        assert stats["win_rate"] == 0
