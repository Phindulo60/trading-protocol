"""Two-way chat with the Telegram bot — users send messages, LLM responds.

Design:
  - Long-polls /getUpdates every ~25s (timeout=25 → server holds connection)
  - Routes messages to:
      • Slash command handlers (/help, /status, /report, /signals, /price)
      • Free-text → Bedrock LLM (Opus → Sonnet fallback) with trading context
  - Keeps last 6 message turns in memory for follow-up questions
  - Filters: only responds in the configured chat_id
  - Runs concurrently with the scan loop via asyncio.create_task

Cold-start behavior: queries Telegram for latest update_id and discards any
queued messages older than ~5 min — avoids replying to stale "test" pings
after a deploy.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Any

import boto3
import httpx

from fsp.notify.telegram import TelegramClient, escape_md

log = logging.getLogger("fsp.chat")

POLL_TIMEOUT = 25            # seconds — long-poll
COLD_START_AGE_LIMIT = 300   # ignore queued messages > 5 min old
HISTORY_LEN = 6              # chat turns kept for follow-up context
DEFAULT_MODEL = "us.anthropic.claude-opus-4-6-v1"
FALLBACK_MODEL = "us.anthropic.claude-sonnet-4-20250514-v1:0"


CHAT_SYSTEM_PROMPT = """You are a forex trading co-pilot embedded in a Telegram bot. The user is a trader who runs an automated USDCAD signal engine. They can ask you anything — about open trades, market state, strategy questions, even general analysis.

## Trading Context You Have
- Current trading session
- Recent signals fired today
- Recent strategy performance (win rate, streak, net R)
- Upcoming economic events (next 4 hours)

## Style
- Keep responses SHORT for Telegram readability — typically 2-4 sentences
- Use plain language, not jargon dumps
- If asked for analysis, be specific and decisive
- If unsure, say so — don't fabricate price levels or stats
- For complex multi-part questions, use bullet points
- NEVER start your reply with "I" or repeat the question

## What You DON'T Do
- You don't execute trades — only the signal engine does
- You don't have live tick data unless given via context
- You don't make up specific price/level numbers — defer to the engine

Respond naturally. If they say "hi" just reply briefly. If they ask "what's USDCAD doing" — use the context provided."""


# ── Helpers ───────────────────────────────────────────────────────────────────

def _format_help() -> str:
    return (
        "🤖 *FSP Trading Bot*\n\n"
        "Just send me a question and I'll reason about it.\n\n"
        "*Commands:*\n"
        "  `/help` — this menu\n"
        "  `/status` — engine health (cycles, today's signals, R)\n"
        "  `/signals` — last 5 signals fired\n"
        "  `/report` — send the daily report now\n"
        "  `/price [PAIR]` — current price (default USDCAD)\n\n"
        "*Examples:*\n"
        "  _What's the trend on USDCAD?_\n"
        "  _Should I worry about the BOC rate decision?_\n"
        "  _How is TREND\\_RSI performing this week?_"
    )


def _format_status_command(cycle_count: int | None, interval_sec: int) -> str:
    """Status: signals today, daily R, recent activity."""
    from fsp.llm.context import signals_today, strategy_stats

    today_sigs = signals_today(pair=None)  # all pairs
    s_trend = strategy_stats("TREND_RSI")
    s_asia = strategy_stats("ASIA_HL")

    lines = [
        f"⚙️ *FSP Engine Status* — {datetime.now(timezone.utc):%H:%M UTC}",
        "",
        f"*Today*: {len(today_sigs)} signal(s) fired",
    ]
    if today_sigs:
        for s in today_sigs[:5]:
            outcome = s.get("outcome") or "open"
            icon = "✅" if outcome == "win" else ("❌" if outcome == "loss" else "🔄")
            lines.append(
                f"  {icon} {s['pair']} {s['strategy']} {s['direction'].upper()} → {outcome}"
            )
    else:
        lines.append("  _no signals today_")

    lines += [
        "",
        f"*TREND\\_RSI* (last {s_trend['total']}): WR {s_trend['win_rate']}%, "
        f"net {s_trend['net_r']:+}R, streak: {s_trend['streak']}",
        f"*ASIA\\_HL* (last {s_asia['total']}): WR {s_asia['win_rate']}%, "
        f"net {s_asia['net_r']:+}R, streak: {s_asia['streak']}",
    ]
    if cycle_count is not None:
        runtime_h = cycle_count * interval_sec / 3600
        lines.append(f"\n🔄 Cycles: {cycle_count} (~{runtime_h:.1f}h)")
    return "\n".join(lines)


def _format_signals_command() -> str:
    """List last 5 signals fired (any outcome)."""
    from fsp.journal.db import conn, migrate
    try:
        with conn() as c:
            migrate(c)
            rows = c.execute(
                "SELECT ts, pair, strategy, direction, entry, outcome, r_multiple "
                "FROM intraday_signals WHERE sent_to_telegram=1 "
                "ORDER BY ts DESC LIMIT 5"
            ).fetchall()
    except Exception as e:
        return f"⚠️ Could not read journal: {escape_md(str(e))}"
    if not rows:
        return "📭 No signals fired yet."

    lines = ["📊 *Last 5 signals*:"]
    for r in rows:
        ts, pair, strat, direction, entry, outcome, rmult = r
        ts_short = ts[5:16].replace("T", " ")
        outcome = outcome or "open"
        icon = "✅" if outcome == "win" else ("❌" if outcome == "loss" else "🔄")
        rstr = f" {rmult:+.2f}R" if rmult is not None else ""
        lines.append(
            f"  {icon} `{ts_short}` {escape_md(strat)} {direction.upper()} "
            f"@ `{entry:.5f}` → {outcome}{rstr}"
        )
    return "\n".join(lines)


def _format_price_command(pair: str = "USDCAD") -> str:
    """Current price + 24h pip change."""
    try:
        from fsp.data.feed import default_feed
        f = default_feed("yf")
        end = datetime.now(timezone.utc)
        df = f.history(pair, "M15", end - timedelta(days=2), end)
        if df.empty or len(df) < 2:
            return f"⚠️ No data for {pair}"
        last = float(df["close"].iloc[-1])
        target = end - timedelta(hours=24)
        prior = df[df.index <= target]
        prior_close = float(prior["close"].iloc[-1]) if not prior.empty else float(df["close"].iloc[0])
        pip = 0.01 if "JPY" in pair else 0.0001
        change_pips = (last - prior_close) / pip
        change_pct = (last - prior_close) / prior_close * 100
        emoji = "📈" if change_pips > 0 else "📉" if change_pips < 0 else "➖"
        return (f"{emoji} *{pair}*: `{last:.5f}`\n"
                f"24h: {change_pips:+.0f} pips ({change_pct:+.2f}%)")
    except Exception as e:
        log.warning("Price fetch failed: %s", e)
        return f"⚠️ Price fetch failed: {escape_md(str(e))}"


# ── Chat handler ──────────────────────────────────────────────────────────────

class ChatHandler:
    """Long-polls Telegram and routes messages to LLM or commands."""

    def __init__(self, tg: TelegramClient, region: str = "us-east-1",
                 cycle_ref: dict | None = None):
        self.tg = tg
        self._br = boto3.client("bedrock-runtime", region_name=region)
        self._model = DEFAULT_MODEL
        self._fallback = FALLBACK_MODEL
        self.offset: int | None = None  # last seen update_id + 1
        self.history: list[dict] = []   # [{role, content}]
        self.cycle_ref = cycle_ref or {}  # mutable dict from live_loop
        self._chat_id_str = str(tg.chat_id)

    async def _get_updates(self) -> list[dict]:
        """Long-poll the Telegram API. Returns new updates."""
        url = f"https://api.telegram.org/bot{self.tg.bot_token}/getUpdates"
        params: dict[str, Any] = {"timeout": POLL_TIMEOUT}
        if self.offset is not None:
            params["offset"] = self.offset
        try:
            async with httpx.AsyncClient(timeout=POLL_TIMEOUT + 10) as c:
                r = await c.get(url, params=params)
                if r.status_code != 200:
                    log.warning("getUpdates HTTP %s: %s", r.status_code, r.text[:200])
                    return []
                data = r.json()
                if not data.get("ok"):
                    log.warning("getUpdates not ok: %s", data)
                    return []
                return data.get("result", [])
        except (httpx.TimeoutException, httpx.RequestError) as e:
            log.debug("getUpdates network blip: %s", e)
            return []

    async def _cold_start(self) -> None:
        """Skip stale messages on container restart."""
        url = f"https://api.telegram.org/bot{self.tg.bot_token}/getUpdates"
        try:
            async with httpx.AsyncClient(timeout=10) as c:
                r = await c.get(url, params={"offset": -1, "timeout": 0})
                if r.status_code == 200:
                    data = r.json()
                    if data.get("ok") and data.get("result"):
                        last = data["result"][-1]
                        self.offset = last["update_id"] + 1
                        log.info("Chat cold-start: skipping queue, offset=%d", self.offset)
        except Exception as e:
            log.warning("Chat cold-start probe failed: %s", e)

    def _is_command(self, text: str) -> bool:
        return text.strip().startswith("/")

    async def _handle_command(self, text: str) -> str:
        """Route slash command to handler. Returns reply string."""
        parts = text.strip().split(maxsplit=1)
        cmd = parts[0].lower().split("@")[0]  # /help@botname → /help
        arg = parts[1] if len(parts) > 1 else ""

        if cmd == "/help" or cmd == "/start":
            return _format_help()
        if cmd == "/status":
            return _format_status_command(
                cycle_count=self.cycle_ref.get("cycle"),
                interval_sec=self.cycle_ref.get("interval_sec", 300),
            )
        if cmd == "/signals":
            return _format_signals_command()
        if cmd == "/price":
            pair = arg.strip().upper() or "USDCAD"
            return _format_price_command(pair)
        if cmd == "/report":
            from fsp.notify.daily_report import compose_report
            try:
                return compose_report(
                    hours=24,
                    pair="USDCAD",
                    cycle_count=self.cycle_ref.get("cycle"),
                    interval_sec=self.cycle_ref.get("interval_sec", 300),
                )
            except Exception as e:
                return f"⚠️ Report failed: {escape_md(str(e))}"
        return f"❓ Unknown command. Try /help"

    def _build_user_context(self) -> str:
        """Build a fresh trading-context preamble for free-text queries."""
        from fsp.llm.context import (
            current_session, upcoming_events,
            signals_today, strategy_stats,
        )
        try:
            sess = current_session()
            events = upcoming_events("USDCAD", hours_ahead=4.0)
            sigs = signals_today(pair=None)
            s_trend = strategy_stats("TREND_RSI")
            s_asia = strategy_stats("ASIA_HL")
        except Exception as e:
            log.warning("Context build failed: %s", e)
            return ""

        parts = [f"## Current Trading Context (auto-injected, {datetime.now(timezone.utc):%H:%M UTC})"]
        parts.append(f"- Session: {sess['name']} — {sess.get('note', '')}")
        if events:
            parts.append(f"- Upcoming events (next 4h):")
            for ev in events[:3]:
                parts.append(f"  • [{ev['impact']}] {ev['event']} ({ev['currency']}) "
                             f"in {ev['minutes_away']} min")
        else:
            parts.append("- No high/medium events in next 4h")
        if sigs:
            parts.append(f"- Today's signals: {len(sigs)}")
            for s in sigs[:5]:
                parts.append(f"  • {s['pair']} {s['strategy']} {s['direction'].upper()} → "
                             f"{s.get('outcome') or 'open'}")
        else:
            parts.append("- No signals fired today")
        parts.append(f"- TREND_RSI last {s_trend['total']}: WR {s_trend['win_rate']}%, "
                     f"net {s_trend['net_r']:+}R")
        parts.append(f"- ASIA_HL last {s_asia['total']}: WR {s_asia['win_rate']}%, "
                     f"net {s_asia['net_r']:+}R")
        return "\n".join(parts)

    async def _ask_llm(self, user_msg: str) -> str:
        """Send free-text to Bedrock with context + history. Returns reply."""
        ctx_preamble = self._build_user_context()
        full_user = f"{ctx_preamble}\n\n## User Message\n{user_msg}" if ctx_preamble else user_msg

        # Build messages: history + new turn
        messages = list(self.history)
        messages.append({"role": "user", "content": [{"text": full_user}]})

        for model_id in [self._model, self._fallback]:
            try:
                resp = self._br.converse(
                    modelId=model_id,
                    messages=messages,
                    system=[{"text": CHAT_SYSTEM_PROMPT}],
                    inferenceConfig={"maxTokens": 600, "temperature": 0.4},
                )
                reply = resp["output"]["message"]["content"][0]["text"].strip()
                # Append to history (use original user_msg, not the context-bloated version)
                self.history.append({"role": "user", "content": [{"text": user_msg}]})
                self.history.append({"role": "assistant", "content": [{"text": reply}]})
                # Trim history
                if len(self.history) > HISTORY_LEN * 2:
                    self.history = self.history[-HISTORY_LEN * 2:]
                return reply
            except Exception as e:
                log.warning("Chat LLM failed (%s) on %s: %s",
                            type(e).__name__, model_id, e)
                if model_id == self._fallback:
                    return f"⚠️ LLM unavailable: `{type(e).__name__}`"
        return "⚠️ Both LLM models failed."

    async def handle_update(self, update: dict) -> None:
        """Process a single Telegram update."""
        msg = update.get("message") or update.get("edited_message") or {}
        if not msg:
            return

        chat = msg.get("chat", {})
        chat_id = str(chat.get("id", ""))
        # Filter: only respond in the bot's configured chat
        if chat_id != self._chat_id_str:
            log.debug("Ignoring message from chat %s (not our chat)", chat_id)
            return

        text = msg.get("text", "").strip()
        if not text:
            return

        # Skip stale messages on cold-start (in case probe missed them)
        msg_date = msg.get("date", 0)
        if msg_date and (time.time() - msg_date) > COLD_START_AGE_LIMIT:
            log.debug("Skipping stale message (age=%.0fs)", time.time() - msg_date)
            return

        log.info("Chat msg: %s", text[:80])

        try:
            if self._is_command(text):
                reply = await self._handle_command(text)
            else:
                reply = await self._ask_llm(text)
        except Exception as e:
            log.exception("Failed to handle message")
            reply = f"⚠️ Error: `{escape_md(str(e))}`"

        # Send reply
        try:
            ok = await self.tg.send(reply)
            if not ok:
                log.warning("Reply send failed; retrying without Markdown")
                # Send plain text as fallback (Markdown parse error is most common cause)
                await self.tg.send(reply, parse_mode="")
        except Exception as e:
            log.exception("Reply send crashed: %s", e)

    async def poll_loop(self) -> None:
        """Forever-loop: long-poll, handle, repeat."""
        await self._cold_start()
        log.info("Chat poll loop started (offset=%s)", self.offset)
        while True:
            try:
                updates = await self._get_updates()
                for upd in updates:
                    self.offset = upd["update_id"] + 1
                    await self.handle_update(upd)
            except asyncio.CancelledError:
                log.info("Chat poll loop cancelled")
                raise
            except Exception as e:
                log.exception("Chat poll iteration failed: %s", e)
                await asyncio.sleep(5)  # back off on unexpected errors
