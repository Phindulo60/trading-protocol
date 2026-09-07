"""Durable send-markers must survive a simulated restart so the calendar brief
and daily report can't spam on a crash/watchdog restart loop."""
import datetime as dt

import pytest

import fsp.journal.db as db
from fsp.journal.db import get_marker, set_marker
from fsp.notify.calendar_report import should_send_calendar


@pytest.fixture(autouse=True)
def _sqlite_backend(tmp_path, monkeypatch):
    # Force the SQLite fallback (not DynamoDB) and isolate the DB file.
    monkeypatch.delenv("FSP_JOURNAL_BACKEND", raising=False)
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "journal.db")
    yield


def test_marker_roundtrip_and_overwrite():
    assert get_marker("calendar_sent") is None          # unset
    set_marker("calendar_sent", "2026-09-07")
    assert get_marker("calendar_sent") == "2026-09-07"   # persisted
    set_marker("calendar_sent", "2026-09-08")
    assert get_marker("calendar_sent") == "2026-09-08"   # overwritten


def test_markers_are_independent():
    set_marker("calendar_sent", "2026-09-07")
    set_marker("daily_report_sent", "2026-09-06")
    assert get_marker("calendar_sent") == "2026-09-07"
    assert get_marker("daily_report_sent") == "2026-09-06"


def test_restart_does_not_resend_calendar():
    """The spam bug: after a send, a fresh boot re-reads the marker and must
    NOT re-fire the brief the same day, even well past the target hour."""
    now = dt.datetime(2026, 9, 7, 20, 41, tzinfo=dt.timezone.utc)  # long after 03:00
    # First boot: nothing sent yet -> due.
    assert should_send_calendar(now, get_marker("calendar_sent")) is True
    # Simulate a successful send persisting the marker.
    set_marker("calendar_sent", now.strftime("%Y-%m-%d"))
    # Simulate a restart: loop re-inits last_calendar_date from durable storage.
    assert should_send_calendar(now, get_marker("calendar_sent")) is False
    # New day -> due again.
    tomorrow = now + dt.timedelta(days=1)
    assert should_send_calendar(tomorrow, get_marker("calendar_sent")) is True
