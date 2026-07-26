"""Tests for the FSP -> mt4-executor execution bridge."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from fsp.execute.bridge import ExecutionConfig, maybe_execute, size_position


@dataclass
class FakeSignal:
    strategy: str = "TREND_RSI"
    pair: str = "EURUSD"
    direction: str = "long"
    entry: float = 1.10000
    sl: float = 1.09800
    tp1: float = 1.10400
    tp2: float = None
    inv_pips: float = 20.0
    rr_tp1: float = 2.0
    rr_tp2: float = None
    risk_r: float = 1.0
    note: str = "test"
    ts: str = "2026-07-26T00:00:00Z"


def _cfg(**kw) -> ExecutionConfig:
    base = dict(
        enabled=True,
        supabase_url="https://x.supabase.co",
        supabase_key="sb_secret_test",
        bot_id="default",
    )
    base.update(kw)
    return ExecutionConfig(**base)


# ── size_position ────────────────────────────────────────────────────────────


def test_size_position_usd_major():
    # $10000 * 0.5% * 1R = $50 risk; 20 pips * $10/pip/lot = $200/lot -> 0.25,
    # capped to default max_lot 0.10.
    lots = size_position(equity=10_000, risk_pct=0.005, risk_r=1.0,
                         inv_pips=20, symbol="EURUSD")
    assert lots == 0.10  # cap


def test_size_position_uncapped_snaps_down():
    # $600 * 0.5% * 1R = $3 risk; $200/lot -> 0.015 -> snap down to 0.01.
    lots = size_position(equity=600, risk_pct=0.005, risk_r=1.0,
                         inv_pips=20, symbol="EURUSD")
    assert lots == 0.01


def test_size_position_reduce_r_halves():
    full = size_position(equity=8000, risk_pct=0.005, risk_r=1.0,
                        inv_pips=20, symbol="EURUSD")
    half = size_position(equity=8000, risk_pct=0.005, risk_r=0.5,
                        inv_pips=20, symbol="EURUSD")
    assert half <= full


def test_size_position_below_min_returns_zero():
    # Tiny equity -> lot rounds below min_lot -> skip.
    lots = size_position(equity=50, risk_pct=0.005, risk_r=1.0,
                        inv_pips=20, symbol="EURUSD")
    assert lots == 0.0


def test_size_position_jpy_pip_size():
    # JPY pip size is 0.01, so loss/lot is 100x a 4-decimal pair for same pips.
    lots = size_position(equity=1_000_000, risk_pct=0.005, risk_r=1.0,
                        inv_pips=20, symbol="USDJPY", max_lot=100)
    # 5000 risk / (20 * 0.01 * 100000 = 20000) = 0.25
    assert lots == 0.25


def test_size_position_guards():
    assert size_position(equity=0, risk_pct=0.005, risk_r=1, inv_pips=20, symbol="EURUSD") == 0.0
    assert size_position(equity=1000, risk_pct=0, risk_r=1, inv_pips=20, symbol="EURUSD") == 0.0
    assert size_position(equity=1000, risk_pct=0.005, risk_r=1, inv_pips=0, symbol="EURUSD") == 0.0


# ── ExecutionConfig ──────────────────────────────────────────────────────────


def test_config_from_env_disabled_by_default(monkeypatch):
    for k in ("FSP_EXECUTE", "SUPABASE_URL", "SUPABASE_SERVICE_KEY", "FSP_EXECUTE_STRATEGIES"):
        monkeypatch.delenv(k, raising=False)
    cfg = ExecutionConfig.from_env()
    assert cfg.enabled is False
    assert cfg.ready is False


def test_config_ready_requires_creds(monkeypatch):
    monkeypatch.setenv("FSP_EXECUTE", "1")
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_SERVICE_KEY", raising=False)
    cfg = ExecutionConfig.from_env()
    assert cfg.enabled is True
    assert cfg.ready is False


def test_config_strategies_parsed(monkeypatch):
    monkeypatch.setenv("FSP_EXECUTE", "1")
    monkeypatch.setenv("SUPABASE_URL", "https://x.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_KEY", "sb_secret_x")
    monkeypatch.setenv("FSP_EXECUTE_STRATEGIES", "trend_rsi, asia_hl")
    cfg = ExecutionConfig.from_env()
    assert cfg.ready is True
    assert cfg.strategies == frozenset({"TREND_RSI", "ASIA_HL"})


# ── maybe_execute gating ─────────────────────────────────────────────────────


def _run(coro):
    return asyncio.run(coro)


def test_maybe_execute_disabled_returns_none():
    cfg = _cfg(enabled=False)
    assert _run(maybe_execute(FakeSignal(), 10_000, 0.005, "TAKE", cfg=cfg)) is None


def test_maybe_execute_skips_ict_shadow():
    cfg = _cfg()
    sig = FakeSignal(strategy="ICT_SHADOW")
    assert _run(maybe_execute(sig, 10_000, 0.005, "TAKE", cfg=cfg)) is None


def test_maybe_execute_skips_on_skip_decision():
    cfg = _cfg()
    assert _run(maybe_execute(FakeSignal(), 10_000, 0.005, "SKIP", cfg=cfg)) is None


def test_maybe_execute_respects_strategy_filter():
    cfg = _cfg(strategies=frozenset({"ASIA_HL"}))
    sig = FakeSignal(strategy="TREND_RSI")
    assert _run(maybe_execute(sig, 10_000, 0.005, "TAKE", cfg=cfg)) is None


def test_maybe_execute_skips_zero_volume():
    cfg = _cfg()
    # equity too small -> 0 lots -> no POST.
    assert _run(maybe_execute(FakeSignal(), 50, 0.005, "TAKE", cfg=cfg)) is None


def _mock_post_ctx(status=201, text=""):
    resp = MagicMock()
    resp.status_code = status
    resp.text = text
    client = AsyncMock()
    client.post = AsyncMock(return_value=resp)
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=client)
    ctx.__aexit__ = AsyncMock(return_value=False)
    return ctx, client


def test_maybe_execute_posts_buy_command():
    cfg = _cfg()
    ctx, client = _mock_post_ctx(201)
    with patch("fsp.execute.bridge.httpx.AsyncClient", return_value=ctx):
        detail = _run(maybe_execute(FakeSignal(direction="long"), 600, 0.005, "TAKE", cfg=cfg))
    assert detail is not None and detail.startswith("buy EURUSD")
    client.post.assert_awaited_once()
    _, kwargs = client.post.call_args
    body = kwargs["json"]
    assert body["type"] == "buy"
    assert body["payload"]["symbol"] == "EURUSD"
    assert body["payload"]["volume"] == 0.01
    assert body["payload"]["comment"] == "fsp:TREND_RSI"
    # key must be on apikey header only, never Authorization.
    assert kwargs["headers"]["apikey"] == "sb_secret_test"
    assert "Authorization" not in kwargs["headers"]


def test_maybe_execute_short_maps_to_sell():
    cfg = _cfg()
    ctx, client = _mock_post_ctx(201)
    with patch("fsp.execute.bridge.httpx.AsyncClient", return_value=ctx):
        _run(maybe_execute(FakeSignal(direction="short"), 600, 0.005, "TAKE", cfg=cfg))
    assert client.post.call_args.kwargs["json"]["type"] == "sell"


def test_maybe_execute_reduce_halves_risk():
    cfg = _cfg(max_lot=100)  # uncapped so the halving is visible
    ctx, client = _mock_post_ctx(201)
    with patch("fsp.execute.bridge.httpx.AsyncClient", return_value=ctx):
        _run(maybe_execute(FakeSignal(), 100_000, 0.005, "REDUCE", cfg=cfg))
    vol_reduce = client.post.call_args.kwargs["json"]["payload"]["volume"]

    ctx2, client2 = _mock_post_ctx(201)
    with patch("fsp.execute.bridge.httpx.AsyncClient", return_value=ctx2):
        _run(maybe_execute(FakeSignal(), 100_000, 0.005, "TAKE", cfg=cfg))
    vol_take = client2.post.call_args.kwargs["json"]["payload"]["volume"]
    assert vol_reduce < vol_take


def test_maybe_execute_swallows_http_error():
    cfg = _cfg()
    ctx, _ = _mock_post_ctx(500, "boom")
    with patch("fsp.execute.bridge.httpx.AsyncClient", return_value=ctx):
        assert _run(maybe_execute(FakeSignal(), 600, 0.005, "TAKE", cfg=cfg)) is None


def test_maybe_execute_swallows_exception():
    cfg = _cfg()
    with patch("fsp.execute.bridge.httpx.AsyncClient", side_effect=RuntimeError("net down")):
        assert _run(maybe_execute(FakeSignal(), 600, 0.005, "TAKE", cfg=cfg)) is None
