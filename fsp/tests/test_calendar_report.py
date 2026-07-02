"""Tests for the daily economic-calendar brief (pure builder + cadence)."""
from __future__ import annotations

from datetime import datetime, timezone

from fsp.notify.calendar_report import (
    build_calendar_brief, should_send_calendar, _arrow,
)

NOW = datetime(2026, 7, 2, 3, 0, tzinfo=timezone.utc)   # 05:00 SAST


def ev(**kw):
    d = {
        "importance": 1, "country": "US", "indicator": "Non-Farm Payrolls",
        "title": "Non-Farm Payrolls", "period": "Jun",
        "date": "2026-07-02T12:30:00.000Z",
        "forecast": "180K", "actual": None, "previous": "175K",
        "forecastRaw": 180000.0, "actualRaw": None, "previousRaw": 175000.0,
    }
    d.update(kw)
    return d


def test_should_send_calendar():
    assert should_send_calendar(NOW, None, target_hour_utc=3) is True
    # already sent today
    assert should_send_calendar(NOW, "2026-07-02", target_hour_utc=3) is False
    # before the target hour
    early = NOW.replace(hour=2)
    assert should_send_calendar(early, None, target_hour_utc=3) is False


def test_arrow():
    assert _arrow(1.4, 0.5) == "\u25B2"   # beat
    assert _arrow(0.2, 0.5) == "\u25BC"   # miss
    assert _arrow(0.5, 0.5) == "="
    assert _arrow(None, 0.5) == ""        # non-numeric


def test_build_brief_filters_and_formats():
    events = [
        ev(),                                                   # today, high, USD
        ev(importance=0, title="Medium Thing"),                 # medium -> excluded
        ev(country="CN", indicator="CN GDP", title="China GDP"),# non-traded -> excluded
        # three historical NFP releases (actual populated, earlier dates)
        ev(date="2026-06-06T12:30:00.000Z", period="May", actual="190K", actualRaw=190000.0, forecast="185K", forecastRaw=185000.0),
        ev(date="2026-05-02T12:30:00.000Z", period="Apr", actual="150K", actualRaw=150000.0, forecast="170K", forecastRaw=170000.0),
        ev(date="2026-04-04T12:30:00.000Z", period="Mar", actual="160K", actualRaw=160000.0, forecast="165K", forecastRaw=165000.0),
    ]
    msg = build_calendar_brief(events, NOW)
    assert msg is not None
    assert "Economic Calendar" in msg
    assert "USD" in msg and "Non-Farm Payrolls" in msg
    assert "180K" in msg                    # today's forecast
    assert "last 3:" in msg
    assert "May 190K" in msg and "Apr 150K" in msg and "Mar 160K" in msg
    assert "Medium Thing" not in msg        # importance filter
    assert "China GDP" not in msg           # currency filter
    # display time = 12:30 UTC -> 14:30 SAST
    assert "14:30" in msg


def test_no_events_today_returns_none():
    # only historical, nothing dated today
    events = [ev(date="2026-06-06T12:30:00.000Z", actual="190K")]
    assert build_calendar_brief(events, NOW) is None


def test_past_three_limited_and_most_recent_first():
    hist_dates = [
        ("2026-06-06T12:30:00.000Z", "May", "190K", 190000.0),
        ("2026-05-02T12:30:00.000Z", "Apr", "150K", 150000.0),
        ("2026-04-04T12:30:00.000Z", "Mar", "160K", 160000.0),
        ("2026-03-07T12:30:00.000Z", "Feb", "140K", 140000.0),
        ("2026-02-07T12:30:00.000Z", "Jan", "130K", 130000.0),
    ]
    events = [ev()] + [ev(date=d, period=p, actual=a, actualRaw=ar) for d, p, a, ar in hist_dates]
    msg = build_calendar_brief(events, NOW)
    line = [l for l in msg.splitlines() if l.strip().startswith("last 3:")][0]
    # only the three most recent, newest first, oldest (Jan/Feb) excluded
    assert line.index("May") < line.index("Apr") < line.index("Mar")
    assert "Feb" not in line and "Jan" not in line
