"""Tests for chat module — slash commands and message routing."""
from __future__ import annotations

import pytest

from fsp.notify.chat import (
    ChatHandler,
    _format_help,
    _format_status_command,
    _format_signals_command,
    _format_price_command,
)


# ── Slash command formatters (no network/LLM) ───────────────────────────────

def test_format_help_basics():
    msg = _format_help()
    assert "FSP Trading Bot" in msg
    assert "/help" in msg
    assert "/status" in msg
    assert "/signals" in msg
    assert "/report" in msg
    assert "/price" in msg


def test_format_status_no_data(tmp_path, monkeypatch):
    monkeypatch.setenv("FSP_DATA_DIR", str(tmp_path))
    msg = _format_status_command(cycle_count=42, interval_sec=300)
    assert "Engine Status" in msg
    assert "0 signal" in msg or "no signals" in msg
    assert "Cycles: 42" in msg


def test_format_status_no_cycle(tmp_path, monkeypatch):
    monkeypatch.setenv("FSP_DATA_DIR", str(tmp_path))
    msg = _format_status_command(cycle_count=None, interval_sec=300)
    # cycle line should be absent
    assert "Cycles:" not in msg


def test_format_signals_empty(tmp_path, monkeypatch):
    monkeypatch.setenv("FSP_DATA_DIR", str(tmp_path))
    monkeypatch.setattr("fsp.journal.db.DB_PATH", tmp_path / "journal.db")
    msg = _format_signals_command()
    assert "No signals" in msg


# ── ChatHandler — instance + command routing ────────────────────────────────

class _StubTG:
    bot_token = "fake_token"
    chat_id = "12345"


def test_chat_handler_construction():
    h = ChatHandler(_StubTG(), cycle_ref={"cycle": 7, "interval_sec": 300})
    assert h.offset is None
    assert h.history == []
    assert h.cycle_ref["cycle"] == 7
    assert h._chat_id_str == "12345"


def test_is_command():
    h = ChatHandler(_StubTG())
    assert h._is_command("/help")
    assert h._is_command("/status")
    assert h._is_command("  /price USDCAD  ")
    assert not h._is_command("hi there")
    assert not h._is_command("what is /help")


@pytest.mark.asyncio
async def test_handle_unknown_command():
    h = ChatHandler(_StubTG())
    reply = await h._handle_command("/wibble")
    assert "Unknown command" in reply


@pytest.mark.asyncio
async def test_handle_help_command():
    h = ChatHandler(_StubTG())
    reply = await h._handle_command("/help")
    assert "FSP Trading Bot" in reply


@pytest.mark.asyncio
async def test_handle_help_with_botname_suffix():
    h = ChatHandler(_StubTG())
    reply = await h._handle_command("/help@fsp_bot")
    assert "FSP Trading Bot" in reply


@pytest.mark.asyncio
async def test_handle_status_command(tmp_path, monkeypatch):
    monkeypatch.setenv("FSP_DATA_DIR", str(tmp_path))
    h = ChatHandler(_StubTG(), cycle_ref={"cycle": 100, "interval_sec": 300})
    reply = await h._handle_command("/status")
    assert "Engine Status" in reply
    assert "Cycles: 100" in reply


@pytest.mark.asyncio
async def test_handle_signals_command(tmp_path, monkeypatch):
    monkeypatch.setenv("FSP_DATA_DIR", str(tmp_path))
    monkeypatch.setattr("fsp.journal.db.DB_PATH", tmp_path / "journal.db")
    h = ChatHandler(_StubTG())
    reply = await h._handle_command("/signals")
    assert "No signals" in reply


# ── Update routing — chat_id filter ─────────────────────────────────────────

@pytest.mark.asyncio
async def test_handle_update_filters_other_chats(monkeypatch):
    """Messages from non-configured chats are ignored."""
    h = ChatHandler(_StubTG())
    sent = []

    class FakeTG:
        async def send(self, *a, **kw):
            sent.append(a)
            return True
    h.tg = FakeTG()
    h._chat_id_str = "12345"

    # Update from different chat
    await h.handle_update({
        "update_id": 1,
        "message": {
            "chat": {"id": 99999},
            "text": "/help",
            "date": 9999999999,  # future date — won't be filtered as stale
        },
    })
    assert sent == []


@pytest.mark.asyncio
async def test_handle_update_filters_stale_messages(monkeypatch):
    """Messages older than COLD_START_AGE_LIMIT are ignored."""
    import time
    h = ChatHandler(_StubTG())
    sent = []

    class FakeTG:
        bot_token = "fake_token"
        chat_id = "12345"
        async def send(self, *a, **kw):
            sent.append(a)
            return True
    h.tg = FakeTG()
    h._chat_id_str = "12345"

    # Stale message (1 hour old)
    await h.handle_update({
        "update_id": 1,
        "message": {
            "chat": {"id": 12345},
            "text": "/help",
            "date": int(time.time()) - 3600,
        },
    })
    assert sent == []


@pytest.mark.asyncio
async def test_handle_update_routes_command(tmp_path, monkeypatch):
    """Valid command from our chat triggers a reply."""
    import time
    monkeypatch.setenv("FSP_DATA_DIR", str(tmp_path))
    h = ChatHandler(_StubTG())
    sent = []

    class FakeTG:
        bot_token = "fake_token"
        chat_id = "12345"
        async def send(self, text, parse_mode="Markdown"):
            sent.append((text, parse_mode))
            return True
    h.tg = FakeTG()
    h._chat_id_str = "12345"

    await h.handle_update({
        "update_id": 1,
        "message": {
            "chat": {"id": 12345},
            "text": "/help",
            "date": int(time.time()),
        },
    })
    assert len(sent) == 1
    assert "FSP Trading Bot" in sent[0][0]


@pytest.mark.asyncio
async def test_handle_update_ignores_empty_text(monkeypatch):
    h = ChatHandler(_StubTG())
    sent = []

    class FakeTG:
        async def send(self, *a, **kw):
            sent.append(a)
            return True
    h.tg = FakeTG()
    h._chat_id_str = "12345"

    await h.handle_update({
        "update_id": 1,
        "message": {"chat": {"id": 12345}, "text": ""},
    })
    assert sent == []
