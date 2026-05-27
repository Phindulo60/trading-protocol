"""Daily signal report — sends a 24h summary to Telegram.

Pulls from the in-container SQLite journal (intraday_signals table). Refreshes
open signal outcomes via the standard resolver before composing.

Triggered:
  - Automatically by live_loop at REPORT_HOUR_UTC (22:00 UTC = midnight SAST)
  - Manually via `fsp daily-report --send`
"""
from __future__ import annotations

import logging
import sqlite3
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any

from fsp.journal.db import DB_PATH, conn, migrate
from fsp.journal.resolver import resolve_all
from fsp.notify.telegram import TelegramClient

log = logging.getLogger("fsp.daily_report")

REPORT_HOUR_UTC = 22  # 22:00 UTC = 00:00 SAST (1h after NY close)


def _signals_in_window(hours: int = 24) -> list[dict[str, Any]]:
    """Return all intraday signals fired in the last `hours`."""
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
    if not DB_PATH.exists():
        return []
    with conn() as c:
        migrate(c)
        rows = c.execute(
            "SELECT id, ts, pair, strategy, direction, entry, sl, tp1, "
            "rr_tp1, risk_r, sent_to_telegram, outcome, r_multiple, exit_ts "
            "FROM intraday_signals WHERE ts >= ? ORDER BY ts ASC",
            (cutoff,),
        ).fetchall()
    return [
        {
            "id": r[0], "ts": r[1], "pair": r[2], "strategy": r[3],
            "direction": r[4], "entry": r[5], "sl": r[6], "tp1": r[7],
            "rr_tp1": r[8], "risk_r": r[9], "sent": bool(r[10]),
            "outcome": r[11], "r_multiple": r[12], "exit_ts": r[13],
        }
        for r in rows
    ]


def _stats(trades: list[dict]) -> dict[str, Any]:
    """Compute aggregate stats. Counts only sent + resolved trades for WR."""
    if not trades:
        return {"n": 0, "sent": 0, "wins": 0, "losses": 0, "open": 0,
                "total_r": 0.0, "wr": 0.0}
    sent = [t for t in trades if t["sent"]]
    closed = [t for t in sent if t["outcome"] in ("win", "loss")]
    wins = [t for t in closed if t["outcome"] == "win"]
    losses = [t for t in closed if t["outcome"] == "loss"]
    timeouts = [t for t in sent if t["outcome"] == "timeout"]
    open_t = [t for t in sent if t["outcome"] is None]
    rs = [t["r_multiple"] for t in sent if t["r_multiple"] is not None]
    total_r = sum(rs)
    wr = len(wins) / len(closed) if closed else 0.0
    return {
        "n": len(trades),
        "sent": len(sent),
        "wins": len(wins),
        "losses": len(losses),
        "timeouts": len(timeouts),
        "open": len(open_t),
        "closed": len(closed),
        "total_r": total_r,
        "wr": wr,
    }


def _outcome_icon(outcome: str | None, r: float | None) -> str:
    if outcome == "win":
        return "✅"
    if outcome == "loss":
        return "❌"
    if outcome == "timeout":
        if r is not None:
            return "🟢" if r > 0 else "🔴" if r < 0 else "⚪"
        return "⏱"
    return "🔄"  # open


def _fmt_trade_line(t: dict) -> str:
    """Compact one-line trade summary."""
    icon = _outcome_icon(t["outcome"], t["r_multiple"])
    direction = t["direction"].upper()
    entry = t["entry"]
    r = t["r_multiple"]
    if t["outcome"] is None:
        r_str = "OPEN"
    elif r is not None:
        r_str = f"{r:+.2f}R"
    else:
        r_str = t["outcome"] or "?"
    # short ts: HH:MM
    ts_short = t["ts"][11:16] if len(t["ts"]) >= 16 else t["ts"]
    return f"  {icon} {ts_short} {t['strategy']} {direction} @ `{entry:.5f}` → {r_str}"


def _get_pair_close(pair: str = "USDCAD") -> tuple[float | None, float | None]:
    """Return (current_close, daily_change_pips) using yfinance. None on failure."""
    try:
        from fsp.data.feed import default_feed
        f = default_feed("yf")
        end = datetime.now(timezone.utc)
        df = f.history(pair, "M15", end - timedelta(days=2), end)
        if df.empty or len(df) < 2:
            return None, None
        last_close = float(df["close"].iloc[-1])
        # Find close ~24h ago
        target_ts = end - timedelta(hours=24)
        prior = df[df.index <= target_ts]
        if prior.empty:
            prior_close = float(df["close"].iloc[0])
        else:
            prior_close = float(prior["close"].iloc[-1])
        pip = 0.01 if "JPY" in pair else 0.0001
        change_pips = (last_close - prior_close) / pip
        return last_close, change_pips
    except Exception as e:
        log.warning("Failed to get %s close: %s", pair, e)
        return None, None


def compose_report(
    hours: int = 24,
    pair: str = "USDCAD",
    cycle_count: int | None = None,
    interval_sec: int = 300,
) -> str:
    """Build the Markdown report text. Resolves outcomes first."""
    # Refresh outcomes (skips last 3h automatically)
    try:
        resolve_all(verbose=False)
    except Exception as e:
        log.warning("resolve_all failed: %s", e)

    # 24h window
    trades = _signals_in_window(hours=hours)

    # 7-day window for context
    trades_7d = _signals_in_window(hours=168)

    s = _stats(trades)
    s7 = _stats(trades_7d)

    # By strategy
    by_strat: dict[str, list] = defaultdict(list)
    for t in trades:
        if t["sent"]:
            by_strat[t["strategy"]].append(t)
    strat_summary = ", ".join(
        f"{name}: {len(lst)}" for name, lst in sorted(by_strat.items())
    ) or "none"

    # Header
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    lines = [
        f"📊 *FSP Daily Report* — {today}",
        "",
    ]

    # 24h block
    if s["sent"] == 0:
        lines.append(f"_No signals fired in last {hours}h._")
    else:
        r_emoji = "📈" if s["total_r"] > 0 else "📉" if s["total_r"] < 0 else "➖"
        lines += [
            f"*Signals*: {s['sent']} ({strat_summary})",
            f"*Daily R*: {s['total_r']:+.2f}R {r_emoji}",
        ]
        if s["closed"] > 0:
            lines.append(f"*WR*: {s['wr']*100:.0f}% ({s['wins']}/{s['closed']} closed)")
        if s["open"] > 0:
            lines.append(f"*Open*: {s['open']} trade(s) still in market")

    # Trade list (max 10)
    sent_trades = [t for t in trades if t["sent"]]
    if sent_trades:
        lines += ["", "🎯 *Trades:*"]
        for t in sent_trades[-10:]:
            lines.append(_fmt_trade_line(t))

    # 7-day context
    if s7["sent"] > 0:
        lines += [
            "",
            f"📈 *7-day*: {s7['total_r']:+.1f}R · "
            f"WR {s7['wr']*100:.0f}% · n={s7['sent']}",
        ]

    # Health
    health_lines = []
    if cycle_count is not None:
        runtime_hrs = cycle_count * interval_sec / 3600
        expected = int(runtime_hrs * 3600 / interval_sec)
        # cycle_count IS the count, expected = same — so use uptime since loop start
        health_lines.append(f"🔄 Cycles: {cycle_count}")
    close, change = _get_pair_close(pair)
    if close is not None:
        change_str = f"({change:+.0f} pips d/d)" if change is not None else ""
        health_lines.append(f"💱 {pair}: `{close:.5f}` {change_str}")

    if health_lines:
        lines += [""] + health_lines

    return "\n".join(lines)


async def send_daily_report(
    tg: TelegramClient,
    hours: int = 24,
    pair: str = "USDCAD",
    cycle_count: int | None = None,
    interval_sec: int = 300,
) -> bool:
    """Compose and send the daily report. Returns True on success."""
    try:
        msg = compose_report(
            hours=hours, pair=pair,
            cycle_count=cycle_count, interval_sec=interval_sec,
        )
    except Exception as e:
        log.exception("Failed to compose daily report")
        msg = f"⚠️ *FSP Daily Report* — error composing report: `{type(e).__name__}: {e}`"

    return await tg.send(msg)


def should_send_report(
    now: datetime,
    last_report_date: str | None,
    target_hour_utc: int = REPORT_HOUR_UTC,
) -> bool:
    """Decide if it's time to send today's report.

    Triggers when:
      - Current UTC hour is at or past target_hour_utc
      - We haven't already sent today's report
    """
    today = now.strftime("%Y-%m-%d")
    if last_report_date == today:
        return False
    if now.hour < target_hour_utc:
        return False
    return True
