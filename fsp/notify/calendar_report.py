"""Daily economic-calendar brief — high-impact ("red") events for the day.

Sent to Telegram once per day (default 05:00 SAST). For each high-impact event
affecting a traded currency it shows the forecast, the actual (once released)
and the previous three releases, so the desk has the surprise history at a
glance before the session.

Data source: TradingView's public economic-calendar API (keyless). Unlike the
ForexFactory weekly feed used by the LLM validator, it exposes actual values,
an ``importance`` rank (1 = high/red) and an ``indicator`` key that groups
recurring releases — which is what makes the "past three releases" lookup work.
"""
from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timedelta, timezone

import requests

from fsp.notify.telegram import TelegramClient, escape_md

log = logging.getLogger("fsp.calendar")

_TV_URL = "https://economic-calendar.tradingview.com/events"
_HDR = {
    "Origin": "https://www.tradingview.com",
    "Referer": "https://www.tradingview.com/",
    "User-Agent": "Mozilla/5.0",
}

# Send hour in UTC. Default 03:00 UTC = 05:00 SAST (South Africa has no DST).
CALENDAR_HOUR_UTC = int(os.environ.get("FSP_CALENDAR_HOUR_UTC", "3"))
# Display timezone: fixed offset (hours) so we don't depend on tzdata in the
# container. SAST = UTC+2 year-round.
_DISPLAY_TZ = timezone(timedelta(hours=int(os.environ.get("FSP_DISPLAY_TZ_OFFSET", "2"))))
_DISPLAY_TZ_LABEL = os.environ.get("FSP_DISPLAY_TZ_LABEL", "SAST")

# TradingView country code -> currency, restricted to what the desk trades.
_COUNTRY_CCY = {"US": "USD", "EU": "EUR", "GB": "GBP", "JP": "JPY", "CA": "CAD", "AU": "AUD"}
_COUNTRIES = ",".join(_COUNTRY_CCY)
_FLAG = {"USD": "\U0001F1FA\U0001F1F8", "EUR": "\U0001F1EA\U0001F1FA",
         "GBP": "\U0001F1EC\U0001F1E7", "JPY": "\U0001F1EF\U0001F1F5",
         "CAD": "\U0001F1E8\U0001F1E6", "AUD": "\U0001F1E6\U0001F1FA"}

_CHUNK_DAYS = 45          # keep each fetch under the API's ~2000-row cap
_HISTORY_DAYS = 135       # enough for 3 occurrences of a 6-weekly rate decision
_MAX_CHARS = 3800         # Telegram hard limit is 4096


def _fetch_chunk(frm: datetime, to: datetime) -> list[dict]:
    r = requests.get(_TV_URL, params={
        "from": frm.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
        "to": to.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
        "countries": _COUNTRIES,
    }, headers=_HDR, timeout=20)
    r.raise_for_status()
    return r.json().get("result", [])


def fetch_calendar(now: datetime, days_back: int = _HISTORY_DAYS) -> list[dict]:
    """Fetch events for the last ``days_back`` days + today, chunked to stay
    under the API row cap (it returns at most ~2000 rows, oldest-first), deduped
    by event id."""
    end = now + timedelta(days=1)
    cur = now - timedelta(days=days_back)
    seen: set = set()
    out: list[dict] = []
    while cur < end:
        nxt = min(cur + timedelta(days=_CHUNK_DAYS), end)
        try:
            for e in _fetch_chunk(cur, nxt):
                eid = e.get("id")
                if eid in seen:
                    continue
                seen.add(eid)
                out.append(e)
        except Exception as ex:
            log.warning("calendar chunk %s..%s failed: %s", cur.date(), nxt.date(), ex)
        cur = nxt
    return out


def _parse_dt(s: str) -> datetime | None:
    try:
        return datetime.fromisoformat(str(s).replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        return None


def _arrow(actual_raw, forecast_raw) -> str:
    """Beat/miss marker vs forecast, using numeric *Raw fields."""
    try:
        a, f = float(actual_raw), float(forecast_raw)
    except (TypeError, ValueError):
        return ""
    return "\u25B2" if a > f else ("\u25BC" if a < f else "=")


def _val(x) -> str:
    return escape_md(str(x)) if x not in (None, "") else "\u2014"


def build_calendar_brief(events: list[dict], now: datetime) -> str | None:
    """Compose the Telegram brief. Returns None when there are no high-impact
    events for the traded currencies on the display-timezone 'today'."""
    disp_now = now.astimezone(_DISPLAY_TZ)
    today = disp_now.date()

    hist: dict[tuple, list[dict]] = {}   # (country, indicator) -> released events
    todays: list[dict] = []
    for e in events:
        if e.get("importance", -9) < 1:              # high-impact ("red") only
            continue
        if e.get("country") not in _COUNTRY_CCY:
            continue
        dt = _parse_dt(e.get("date", ""))
        if dt is None:
            continue
        rec = {"dt": dt, **e}
        if e.get("actual") is not None:
            hist.setdefault((e.get("country"), e.get("indicator")), []).append(rec)
        if dt.astimezone(_DISPLAY_TZ).date() == today:
            todays.append(rec)

    if not todays:
        return None
    for v in hist.values():
        v.sort(key=lambda x: x["dt"])
    todays.sort(key=lambda x: x["dt"])

    by_ccy: dict[str, list[dict]] = {}
    for e in todays:
        by_ccy.setdefault(_COUNTRY_CCY[e["country"]], []).append(e)

    lines = [
        f"\U0001F4C5 *Economic Calendar* \u2014 {escape_md(disp_now.strftime('%a %d %b %Y'))}",
        f"_High-impact events \u00b7 times {_DISPLAY_TZ_LABEL}_",
        "",
    ]
    for ccy in sorted(by_ccy, key=lambda c: min(x["dt"] for x in by_ccy[c])):
        lines.append(f"{_FLAG.get(ccy, '')} *{ccy}*")
        for e in by_ccy[ccy]:
            t = e["dt"].astimezone(_DISPLAY_TZ).strftime("%H:%M")
            title = escape_md(e.get("title") or e.get("indicator") or "Event")
            period = e.get("period") or ""
            per = f" _{escape_md(period)}_" if period else ""
            lines.append(f"  `{t}`  {title}{per}")
            if any(e.get(k) not in (None, "") for k in ("forecast", "actual", "previous")):
                lines.append(f"      F {_val(e.get('forecast'))} \u00b7 "
                             f"A {_val(e.get('actual'))} \u00b7 P {_val(e.get('previous'))}")
            key = (e["country"], e.get("indicator"))
            past = [h for h in hist.get(key, []) if h["dt"] < e["dt"]][-3:]
            if past:
                chunks = []
                for h in reversed(past):          # most recent first
                    lbl = h.get("period") or h["dt"].strftime("%b")
                    chunks.append(f"{escape_md(str(lbl))} {escape_md(str(h.get('actual')))}"
                                  f"{_arrow(h.get('actualRaw'), h.get('forecastRaw'))}")
                lines.append("      last 3: " + " \u00b7 ".join(chunks))
        lines.append("")

    lines.append("_F=forecast A=actual P=previous \u00b7 \u25B2/\u25BC vs forecast_")
    msg = "\n".join(lines).strip()
    if len(msg) > _MAX_CHARS:
        msg = msg[:_MAX_CHARS].rsplit("\n", 1)[0] + "\n_\u2026 (truncated)_"
    return msg


def should_send_calendar(now: datetime, last_calendar_date: str | None,
                         target_hour_utc: int = CALENDAR_HOUR_UTC) -> bool:
    """True when today's brief is due: current UTC hour >= target and not yet
    sent today."""
    if last_calendar_date == now.strftime("%Y-%m-%d"):
        return False
    return now.hour >= target_hour_utc


def _compose_brief(now: datetime) -> str:
    events = fetch_calendar(now)
    msg = build_calendar_brief(events, now)
    if msg is None:
        d = now.astimezone(_DISPLAY_TZ).strftime("%a %d %b %Y")
        return (f"\U0001F4C5 *Economic Calendar* \u2014 {escape_md(d)}\n"
                f"_No high-impact events for the traded currencies today._")
    return msg


async def send_calendar_brief(tg: TelegramClient, now: datetime | None = None) -> bool:
    """Fetch + compose + send the daily calendar brief. The (blocking) HTTP
    fetch runs in a worker thread so it can't stall the scan loop / watchdog."""
    now = now or datetime.now(timezone.utc)
    try:
        msg = await asyncio.to_thread(_compose_brief, now)
    except Exception as e:
        log.exception("calendar brief failed")
        msg = f"\u26A0\uFE0F *Economic Calendar* \u2014 error: `{type(e).__name__}: {e}`"
    return await tg.send(msg)
