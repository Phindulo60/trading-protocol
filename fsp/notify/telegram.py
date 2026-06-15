"""Telegram notifier — sends formatted setup alerts."""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

import httpx

from fsp.grader.setup import SetupCandidate
from fsp.data.types import Grade

log = logging.getLogger(__name__)


def escape_md(text: str) -> str:
    """Escape Telegram legacy-Markdown special chars in free-text content.

    Use for any LLM-generated or user-supplied text that goes inside an
    already-Markdown-formatted message. Hardcoded markup is unaffected.
    """
    if not text:
        return ""
    # Legacy Markdown special chars: _ * ` [ — backslash first to avoid double-escape
    return (text.replace("\\", "\\\\")
                .replace("_", r"\_")
                .replace("*", r"\*")
                .replace("`", r"\`")
                .replace("[", r"\["))


@dataclass
class TelegramClient:
    """Telegram bot client with multi-recipient fan-out.

    `chat_id` is the *primary* chat (always your DM) — used by the chat
    poller for command authorization. `extra_chat_ids` are additional
    recipients (groups, channels, second person) that receive signals only.

    Sends fan out to all chat_ids in parallel. send() returns True if at
    least one delivery succeeded.
    """
    bot_token: str
    chat_id: str
    extra_chat_ids: list[str] = field(default_factory=list)

    @property
    def all_chat_ids(self) -> list[str]:
        return [self.chat_id] + list(self.extra_chat_ids)

    async def _send_one(self, chat_id: str, text: str, parse_mode: str) -> bool:
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        try:
            async with httpx.AsyncClient(timeout=15) as c:
                r = await c.post(url, json={
                    "chat_id": chat_id,
                    "text": text,
                    "parse_mode": parse_mode,
                    "disable_web_page_preview": True,
                })
                if r.status_code != 200:
                    log.error("Telegram send to %s failed %s: %s",
                              chat_id, r.status_code, r.text[:200])
                    return False
                return True
        except Exception as e:
            log.error("Telegram send to %s exception: %s", chat_id, e)
            return False

    async def send(self, text: str, parse_mode: str = "Markdown") -> bool:
        """Fan out to all configured chat_ids. True if any delivery succeeded."""
        ids = self.all_chat_ids
        if len(ids) == 1:
            # Fast path: skip gather overhead when only one recipient
            return await self._send_one(ids[0], text, parse_mode)
        results = await asyncio.gather(
            *[self._send_one(cid, text, parse_mode) for cid in ids],
            return_exceptions=True,
        )
        ok_count = sum(1 for r in results if r is True)
        if ok_count < len(ids):
            log.warning("Telegram fan-out partial: %d/%d delivered",
                        ok_count, len(ids))
        return ok_count > 0

    def send_sync(self, text: str) -> bool:
        return asyncio.run(self.send(text))


def format_setup(setup: SetupCandidate) -> str:
    icon = {"A+": "🟢", "A": "🟢", "B": "🟡", "SKIP": "🔴"}[setup.grade.value]
    lines = [
        f"{icon} *{setup.grade.value} — {setup.pair} {setup.direction.upper() if setup.direction else '—'}*",
    ]
    if setup.grade != Grade.SKIP and setup.entry is not None:
        lines += [
            f"Entry: `{setup.entry:.5f}`",
            f"SL: `{setup.sl:.5f}`  ({setup.invalidation_pips:.1f}p)",
        ]
        if setup.tp1 is not None:
            lines.append(f"TP1: `{setup.tp1:.5f}`  ({setup.rr_tp1:.1f}R — {setup.context.get('opposing_target')})")
        if setup.tp2 is not None:
            lines.append(f"TP2: `{setup.tp2:.5f}`  ({setup.rr_tp2:.1f}R)")
        lines += [
            f"Risk: *{setup.risk_r:.1f}R*  ·  Key: {setup.key_level_ref}",
        ]
    ctx = setup.context
    lines.append(f"Session: {ctx['session']} · Cycle: {ctx['cycle']} · ADR%: {ctx['adr_pct']} · Bias: {ctx['bias']}")
    lines.append(f"✓ {setup.passed()}/{setup.total()} checks")
    failed = [c.name for c in setup.checklist if not c.passed]
    if failed:
        lines.append("_Missing:_ " + ", ".join(failed))
    return "\n".join(lines)


async def get_updates_chat_id(bot_token: str) -> list[tuple[str, str]]:
    """Return list of (chat_id, chat_title) from recent updates. Used by setup wizard."""
    url = f"https://api.telegram.org/bot{bot_token}/getUpdates"
    async with httpx.AsyncClient(timeout=15) as c:
        r = await c.get(url)
        r.raise_for_status()
        data = r.json()
    chats: dict[str, str] = {}
    for upd in data.get("result", []):
        msg = upd.get("message") or upd.get("edited_message") or upd.get("channel_post") or {}
        chat = msg.get("chat")
        if chat:
            cid = str(chat["id"])
            title = chat.get("title") or chat.get("username") or chat.get("first_name") or cid
            chats[cid] = title
    return list(chats.items())


def format_signal(sig) -> str:
    """Format an intraday Signal for Telegram. Routes to strategy-specific formatters."""
    if sig.strategy == "LEVEL_OB":
        return _format_level_ob(sig)
    return _format_generic(sig)


def _format_level_ob(sig) -> str:
    """Rich formatter for LEVEL_OB — shows tier, level, OB range, hist stats."""
    ctx   = sig.context
    tier  = ctx.get("tier", "?")
    lname = ctx.get("level_type", "?")
    lp    = ctx.get("level_price", 0.0)

    tier_meta = {
        "CONF": ("\u2b50", "CONFLUENCE  M15+H1 OB", "100% hist rev, avg 76p, n=12"),
        "H1":   ("\U0001f535", "H1 OB",             "87% hist rev, avg 58p, n=194"),
        "M15":  ("\U0001f7e1", "M15 OB",             "81% hist rev, avg 51p, n=109"),
    }.get(tier, ("\U0001f535", tier, ""))
    t_icon, t_label, t_stats = tier_meta

    dir_icon = "\U0001f4c8" if sig.direction == "long" else "\U0001f4c9"
    ob_range = ctx.get("ob_range", [0, 0])
    ob_lo, ob_hi = ob_range[0], ob_range[1]

    lines = [
        f"{t_icon}{dir_icon} *[LEVEL OB] {sig.pair} {sig.direction.upper()}*",
        f"Tier:   *{t_label}*  _{t_stats}_",
        f"Level:  `{lname}` @ `{lp:.5f}`  |  Session: {ctx.get('session','?')}  |  H4: {ctx.get('h4_trend','?')}",
        f"OB:     `{ob_lo:.5f}` - `{ob_hi:.5f}`",
        f"",
        f"Entry:  `{sig.entry:.5f}`",
        f"SL:     `{sig.sl:.5f}`  ({sig.inv_pips:.1f} pips)",
        f"TP1:    `{sig.tp1:.5f}`  ({sig.rr_tp1:.1f}R)",
    ]
    if sig.tp2 is not None:
        lines.append(f"TP2:    `{sig.tp2:.5f}`  ({sig.rr_tp2:.1f}R)")
    lines.append(f"Risk:   *{sig.risk_r:.1f}R*  |  Max hold 8 bars (~2h)")
    return "\n".join(lines)


def _format_generic(sig) -> str:
    """Formatter for TREND RSI / ECM / ARB signals."""
    icon = "\U0001f535" if sig.direction == "long" else "\U0001f7e0"
    strat_label = {
        "ECM":       "EMA Cross Momentum",
        "ARB":       "Asian Range Breakout",
        "TREND_RSI": "Trend RSI",
    }.get(sig.strategy, sig.strategy.replace("_", " "))
    s_display = sig.strategy.replace("_", " ")
    lines = [
        f"{icon} *[{s_display}] {sig.pair} {sig.direction.upper()}*  _{strat_label}_",
        f"Entry:  `{sig.entry:.5f}`",
        f"SL:     `{sig.sl:.5f}`  ({sig.inv_pips:.1f} pips)",
        f"TP1:    `{sig.tp1:.5f}`  ({sig.rr_tp1:.1f}R)",
    ]
    if sig.tp2 is not None:
        lines.append(f"TP2:    `{sig.tp2:.5f}`  ({sig.rr_tp2:.1f}R)" if sig.rr_tp2 else
                     f"TP2:    `{sig.tp2:.5f}`")
    lines += [f"Risk:   *{sig.risk_r:.1f}R*", f"_{sig.note}_"]
    ctx = sig.context
    extras = []
    if "session" in ctx: extras.append(f"Session: {ctx['session']}")
    if "rsi"     in ctx: extras.append(f"RSI: {ctx['rsi']}")
    if extras:
        lines.append(" \u00b7 ".join(extras))
    return "\n".join(lines)
