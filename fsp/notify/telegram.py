"""Telegram notifier — sends formatted setup alerts."""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

import httpx

from fsp.grader.setup import SetupCandidate
from fsp.data.types import Grade

log = logging.getLogger(__name__)


@dataclass
class TelegramClient:
    bot_token: str
    chat_id: str

    async def send(self, text: str, parse_mode: str = "Markdown") -> bool:
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        async with httpx.AsyncClient(timeout=15) as c:
            r = await c.post(url, json={
                "chat_id": self.chat_id,
                "text": text,
                "parse_mode": parse_mode,
                "disable_web_page_preview": True,
            })
            if r.status_code != 200:
                log.error("Telegram send failed %s: %s", r.status_code, r.text)
                return False
            return True

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


def format_signal(sig: "Signal") -> str:  # type: ignore[name-defined]
    """Format an intraday Signal (ECM/ARB) for Telegram."""
    from fsp.signals.base import Signal as _Signal
    icon = "🔵" if sig.direction == "long" else "🟠"
    strat_label = {
        "ECM": "EMA Cross Momentum",
        "ARB": "Asian Range Breakout",
        "TREND_RSI": "Trend RSI",
    }.get(sig.strategy, sig.strategy.replace("_", " "))
    lines = [
        f"{icon} *[{sig.strategy}] {sig.pair} {sig.direction.upper()}*  _{strat_label}_",
        f"Entry:  `{sig.entry:.5f}`",
        f"SL:     `{sig.sl:.5f}`  ({sig.inv_pips:.1f} pips)",
        f"TP1:    `{sig.tp1:.5f}`  ({sig.rr_tp1:.1f}R)",
    ]
    if sig.tp2 is not None:
        lines.append(f"TP2:    `{sig.tp2:.5f}`  ({sig.rr_tp2:.1f}R)" if sig.rr_tp2 else
                     f"TP2:    `{sig.tp2:.5f}`")
    lines += [
        f"Risk:   *{sig.risk_r:.1f}R*",
        f"_{sig.note}_",
    ]
    ctx = sig.context
    extras = []
    if "session" in ctx:
        extras.append(f"Session: {ctx['session']}")
    if "rsi" in ctx:
        extras.append(f"RSI: {ctx['rsi']}")
    if "adr_pct" in ctx:
        extras.append(f"ADR%: {ctx['adr_pct']}")
    if extras:
        lines.append(" · ".join(extras))
    return "\n".join(lines)
