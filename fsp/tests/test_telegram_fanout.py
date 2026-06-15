"""Tests for multi-recipient Telegram fan-out + chat_id parsing."""
from __future__ import annotations

import asyncio
from unittest.mock import patch, AsyncMock, MagicMock

import pytest

from fsp.notify.config import parse_chat_ids
from fsp.notify.telegram import TelegramClient


# ── parse_chat_ids ──────────────────────────────────────────────────────────


def test_parse_single_chat_id():
    primary, extras = parse_chat_ids("12345")
    assert primary == "12345"
    assert extras == []


def test_parse_multiple_chat_ids():
    primary, extras = parse_chat_ids("12345,-67890,99999")
    assert primary == "12345"
    assert extras == ["-67890", "99999"]


def test_parse_strips_whitespace():
    primary, extras = parse_chat_ids("  12345 , -67890 ,  99999  ")
    assert primary == "12345"
    assert extras == ["-67890", "99999"]


def test_parse_negative_group_id():
    """Group chat_ids are negative; must be preserved as strings."""
    primary, extras = parse_chat_ids("5336135541,-5370235404")
    assert primary == "5336135541"
    assert extras == ["-5370235404"]


def test_parse_empty_raises():
    with pytest.raises(ValueError):
        parse_chat_ids("")
    with pytest.raises(ValueError):
        parse_chat_ids(",,,")


def test_parse_skips_empty_segments():
    """Trailing/leading commas should not produce empty entries."""
    primary, extras = parse_chat_ids("12345,,67890,")
    assert primary == "12345"
    assert extras == ["67890"]


# ── TelegramClient fan-out ──────────────────────────────────────────────────


def test_client_default_no_extras():
    tg = TelegramClient(bot_token="fake", chat_id="100")
    assert tg.all_chat_ids == ["100"]


def test_client_with_extras():
    tg = TelegramClient(bot_token="fake", chat_id="100",
                        extra_chat_ids=["-200", "300"])
    assert tg.all_chat_ids == ["100", "-200", "300"]


def test_extras_default_isolated():
    """Each instance must get its own extras list (no shared mutable default)."""
    a = TelegramClient(bot_token="fake", chat_id="1")
    b = TelegramClient(bot_token="fake", chat_id="2")
    a.extra_chat_ids.append("xyz")
    assert b.extra_chat_ids == []  # isolated


@pytest.mark.asyncio
async def test_send_single_recipient():
    """Single chat_id: should call _send_one once and return its result."""
    tg = TelegramClient(bot_token="fake", chat_id="100")
    with patch.object(tg, "_send_one", new=AsyncMock(return_value=True)) as mock:
        result = await tg.send("hello")
        assert result is True
        mock.assert_called_once_with("100", "hello", "Markdown")


@pytest.mark.asyncio
async def test_send_fan_out_all_succeed():
    """Multi-recipient: all chat_ids called, returns True when all succeed."""
    tg = TelegramClient(bot_token="fake", chat_id="100",
                        extra_chat_ids=["-200", "300"])
    with patch.object(tg, "_send_one", new=AsyncMock(return_value=True)) as mock:
        result = await tg.send("hello")
        assert result is True
        assert mock.call_count == 3
        called_ids = [c.args[0] for c in mock.call_args_list]
        assert set(called_ids) == {"100", "-200", "300"}


@pytest.mark.asyncio
async def test_send_fan_out_partial_failure():
    """If one recipient fails but others succeed, send() still returns True."""
    tg = TelegramClient(bot_token="fake", chat_id="100",
                        extra_chat_ids=["-200"])
    # First (primary) succeeds, second (group) fails
    side_effects = [True, False]
    with patch.object(tg, "_send_one", new=AsyncMock(side_effect=side_effects)):
        result = await tg.send("hello")
        assert result is True  # at least one delivered


@pytest.mark.asyncio
async def test_send_fan_out_all_fail():
    """If all recipients fail, send() returns False."""
    tg = TelegramClient(bot_token="fake", chat_id="100",
                        extra_chat_ids=["-200"])
    with patch.object(tg, "_send_one", new=AsyncMock(return_value=False)):
        result = await tg.send("hello")
        assert result is False


@pytest.mark.asyncio
async def test_send_one_handles_http_error():
    """_send_one must return False on non-200, not raise."""
    tg = TelegramClient(bot_token="fake", chat_id="100")
    fake_response = MagicMock()
    fake_response.status_code = 403
    fake_response.text = "Bot was blocked by the user"

    with patch("fsp.notify.telegram.httpx.AsyncClient") as mock_client:
        mock_ctx = AsyncMock()
        mock_ctx.__aenter__.return_value.post = AsyncMock(return_value=fake_response)
        mock_client.return_value = mock_ctx
        result = await tg._send_one("100", "hi", "Markdown")
        assert result is False


@pytest.mark.asyncio
async def test_send_one_handles_network_exception():
    """_send_one must return False on network exception, not raise."""
    import httpx
    tg = TelegramClient(bot_token="fake", chat_id="100")

    with patch("fsp.notify.telegram.httpx.AsyncClient") as mock_client:
        mock_ctx = AsyncMock()
        mock_ctx.__aenter__.return_value.post = AsyncMock(
            side_effect=httpx.ConnectError("network down")
        )
        mock_client.return_value = mock_ctx
        result = await tg._send_one("100", "hi", "Markdown")
        assert result is False


@pytest.mark.asyncio
async def test_one_recipient_failure_does_not_block_others():
    """Concurrent fan-out: a slow/failed recipient shouldnt delay others.

    Use staggered sleep + one failure to ensure return_exceptions handles it.
    """
    tg = TelegramClient(bot_token="fake", chat_id="100",
                        extra_chat_ids=["-200", "300"])

    async def fake_send(cid, *a, **kw):
        if cid == "-200":
            raise RuntimeError("boom")
        await asyncio.sleep(0.01)
        return True

    with patch.object(tg, "_send_one", side_effect=fake_send):
        result = await tg.send("hi")
        # Two of three succeeded → True (any-success rule)
        assert result is True
