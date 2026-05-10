"""Live market context builder — gathers session, calendar, and journal data for the LLM."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import requests

log = logging.getLogger(__name__)

# ── Session Detection ─────────────────────────────────────────────────────────

# All times in UTC
_SESSIONS = [
    ("ASIA",  0,  8),   # 00:00 – 08:00 UTC  (Tokyo/Sydney)
    ("LO",    8, 12),   # 08:00 – 12:00 UTC  (London morning)
    ("NY-AM", 12, 16),  # 12:00 – 16:00 UTC  (NY morning / LO-NY overlap 12-16)
    ("NY-PM", 16, 21),  # 16:00 – 21:00 UTC  (NY afternoon)
    ("OFF",   21, 24),  # 21:00 – 00:00 UTC  (after hours)
]

# Pairs sensitive to specific sessions
_SESSION_NOTES = {
    "ASIA":  "Low vol for EUR/GBP majors. JPY pairs most active.",
    "LO":    "Peak vol for EUR/GBP. Best session for breakouts.",
    "NY-AM": "LO-NY overlap = highest liquidity window of the day.",
    "NY-PM": "Vol declining. Avoid new entries after 19:00 UTC.",
    "OFF":   "Market closing. Very low liquidity — avoid trading.",
}


def current_session() -> dict:
    """Return current session name + trading notes."""
    now = datetime.now(timezone.utc)
    hour = now.hour
    for name, start, end in _SESSIONS:
        if start <= hour < end:
            return {
                "name": name,
                "time": now.strftime("%H:%M UTC (%A)"),
                "note": _SESSION_NOTES.get(name, ""),
            }
    return {"name": "OFF", "time": now.strftime("%H:%M UTC (%A)"), "note": ""}


# ── Economic Calendar ─────────────────────────────────────────────────────────

_FF_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"

# Map forex factory country codes to currencies we care about
_PAIR_CURRENCIES = {
    "EURUSD": ["EUR", "USD"],
    "GBPUSD": ["GBP", "USD"],
    "AUDUSD": ["AUD", "USD"],
    "USDCAD": ["USD", "CAD"],
    "EURJPY": ["EUR", "JPY"],
    "GBPJPY": ["GBP", "JPY"],
    "USDJPY": ["USD", "JPY"],
}

_cached_calendar: tuple[float, list[dict]] | None = None


def _fetch_calendar() -> list[dict]:
    """Fetch and cache the weekly calendar (refresh every 30 min)."""
    global _cached_calendar
    import time
    if _cached_calendar:
        ts, data = _cached_calendar
        if time.time() - ts < 1800:  # 30-min cache
            return data

    try:
        resp = requests.get(_FF_URL, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        _cached_calendar = (time.time(), data)
        return data
    except Exception as e:
        log.warning("Calendar fetch failed: %s", e)
        return _cached_calendar[1] if _cached_calendar else []


def upcoming_events(pair: str, hours_ahead: float = 4.0) -> list[dict]:
    """Return upcoming high/medium impact events relevant to a pair."""
    from dateutil import parser as dtparser

    calendar = _fetch_calendar()
    if not calendar:
        return []

    currencies = set(_PAIR_CURRENCIES.get(pair, []))
    if not currencies:
        currencies = {"USD"}  # fallback

    now = datetime.now(timezone.utc)
    cutoff = now + timedelta(hours=hours_ahead)
    lookback = now - timedelta(hours=1)  # include events in last hour (just happened)

    relevant = []
    for ev in calendar:
        if ev.get("impact", "").lower() not in ("high", "medium"):
            continue
        if ev.get("country", "") not in currencies:
            continue
        try:
            ev_time = dtparser.parse(ev["date"])
            if ev_time.tzinfo is None:
                ev_time = ev_time.replace(tzinfo=timezone.utc)
            ev_time = ev_time.astimezone(timezone.utc)
        except (ValueError, KeyError):
            continue

        if lookback <= ev_time <= cutoff:
            minutes_away = (ev_time - now).total_seconds() / 60
            relevant.append({
                "event": ev.get("title", "Unknown"),
                "currency": ev.get("country", "?"),
                "time": ev_time.strftime("%H:%M UTC"),
                "impact": ev.get("impact", "?"),
                "forecast": ev.get("forecast", ""),
                "previous": ev.get("previous", ""),
                "minutes_away": round(minutes_away),
            })

    return sorted(relevant, key=lambda x: x["minutes_away"])


# ── Journal / Recent Trades ───────────────────────────────────────────────────

def recent_trades(pair: str | None = None, strategy: str | None = None,
                  limit: int = 10) -> list[dict]:
    """Pull recent resolved trades from journal DB."""
    try:
        from fsp.journal.db import conn, migrate
        with conn() as c:
            migrate(c)
            query = ("SELECT pair, strategy, direction, outcome, r_multiple, ts, exit_ts "
                     "FROM intraday_signals WHERE outcome IS NOT NULL ")
            params: list = []
            if pair:
                query += "AND pair=? "
                params.append(pair)
            if strategy:
                query += "AND strategy=? "
                params.append(strategy)
            query += "ORDER BY ts DESC LIMIT ?"
            params.append(limit)
            rows = c.execute(query, params).fetchall()
    except Exception as e:
        log.warning("Journal query failed: %s", e)
        return []

    return [
        {"pair": r[0], "strategy": r[1], "direction": r[2],
         "outcome": r[3], "r": r[4], "ts": r[5], "exit_ts": r[6]}
        for r in rows
    ]


def strategy_stats(strategy: str, last_n: int = 20) -> dict:
    """Compute recent win rate and streak for a strategy."""
    trades = recent_trades(strategy=strategy, limit=last_n)
    if not trades:
        return {"total": 0, "win_rate": 0, "streak": "none", "net_r": 0}

    wins = sum(1 for t in trades if t["outcome"] == "win")
    total = len(trades)
    net_r = sum(t.get("r", 0) or 0 for t in trades)

    # Current streak (from most recent)
    streak_type = trades[0]["outcome"] if trades else "none"
    streak_count = 0
    for t in trades:
        if t["outcome"] == streak_type:
            streak_count += 1
        else:
            break

    return {
        "total": total,
        "win_rate": round(wins / total * 100, 1) if total else 0,
        "streak": f"{streak_count} {streak_type}{'s' if streak_count > 1 else ''}",
        "net_r": round(net_r, 1),
    }


def signals_today(pair: str | None = None) -> list[dict]:
    """Get all signals fired today (for correlation/exposure check)."""
    try:
        from fsp.journal.db import conn
        today_start = datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0).isoformat()
        with conn() as c:
            query = ("SELECT pair, strategy, direction, entry, outcome, sent_to_telegram "
                     "FROM intraday_signals WHERE ts >= ? AND sent_to_telegram = 1 ")
            params: list = [today_start]
            if pair:
                query += "AND pair != ? "  # exclude current pair — we want OTHER open signals
                params.append(pair)
            query += "ORDER BY ts DESC"
            rows = c.execute(query, params).fetchall()
    except Exception as e:
        log.warning("Today's signals query failed: %s", e)
        return []

    return [
        {"pair": r[0], "strategy": r[1], "direction": r[2],
         "entry": r[3], "outcome": r[4], "sent": bool(r[5])}
        for r in rows
    ]


# ── Price Context ─────────────────────────────────────────────────────────────

def price_context(pair: str, m15_df, h1_df=None) -> dict:
    """Extract key technical context from price data."""
    ctx = {}
    if m15_df is not None and len(m15_df) >= 20:
        close = m15_df["close"]
        ctx["last_price"] = round(float(close.iloc[-1]), 5)
        # Simple EMA20 trend on M15
        ema20 = close.ewm(span=20).mean()
        ctx["m15_trend"] = "bullish" if float(close.iloc[-1]) > float(ema20.iloc[-1]) else "bearish"
        # ATR for volatility context
        high = m15_df["high"]
        low = m15_df["low"]
        tr = (high - low).rolling(14).mean()
        if len(tr.dropna()) > 0:
            ctx["m15_atr"] = round(float(tr.iloc[-1]), 5)

    if h1_df is not None and len(h1_df) >= 20:
        close_h1 = h1_df["close"]
        ema20_h1 = close_h1.ewm(span=20).mean()
        ctx["h1_trend"] = "bullish" if float(close_h1.iloc[-1]) > float(ema20_h1.iloc[-1]) else "bearish"
        # Daily range
        if len(h1_df) >= 2:
            today_bars = h1_df.tail(6)  # last ~6 H1 bars
            ctx["today_range"] = round(float(today_bars["high"].max() - today_bars["low"].min()), 5)

    return ctx


# ── Full Context Builder ──────────────────────────────────────────────────────

def build_context(pair: str, strategy: str, m15_df=None, h1_df=None) -> dict:
    """Build complete market context for the LLM analyst."""
    return {
        "session": current_session(),
        "upcoming_events": upcoming_events(pair),
        "recent_trades": recent_trades(pair=pair, strategy=strategy, limit=8),
        "strategy_stats": strategy_stats(strategy),
        "signals_today": signals_today(pair),
        "price": price_context(pair, m15_df, h1_df),
    }
