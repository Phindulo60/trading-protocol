"""Tests for the DynamoDB journal backend (mocked with moto)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone

import pytest

pytest.importorskip("moto")
import boto3
from moto import mock_aws


# ── Minimal signal stand-ins ─────────────────────────────────────────────────

@dataclass
class FakeSignal:
    ts: str
    pair: str
    strategy: str
    direction: str
    entry: float
    sl: float
    tp1: float
    tp2: float
    inv_pips: float
    rr_tp1: float
    risk_r: float
    note: str
    context: dict = field(default_factory=dict)


@dataclass
class FakeCheck:
    name: str
    passed: bool
    note: str = ""


@dataclass
class FakeGrade:
    value: str


@dataclass
class FakeSetup:
    pair: str
    grade: FakeGrade
    direction: str
    entry: float
    sl: float
    tp1: float
    tp2: float
    risk_r: float
    checklist: list = field(default_factory=list)
    context: dict = field(default_factory=dict)


@pytest.fixture
def dynamo_journal():
    """Spin up a mocked DynamoDB table + DynamoJournal instance."""
    with mock_aws():
        os.environ["AWS_DEFAULT_REGION"] = "us-east-1"
        from fsp.journal.dynamo import DynamoJournal, ensure_table
        ensure_table("fsp-journal-test", region="us-east-1")
        yield DynamoJournal(table_name="fsp-journal-test", region="us-east-1")


def _mk_signal(ts="2026-06-17T08:00:00+00:00", pair="EURUSD",
               strategy="TREND_RSI", direction="long"):
    return FakeSignal(
        ts=ts, pair=pair, strategy=strategy, direction=direction,
        entry=1.1000, sl=1.0950, tp1=1.1150, tp2=1.1200,
        inv_pips=50.0, rr_tp1=3.0, risk_r=1.0, note="test",
        context={"session": "LONDON", "adr_pct": 0.45},
    )


# ── Provisioning ─────────────────────────────────────────────────────────────

def test_ensure_table_idempotent():
    with mock_aws():
        from fsp.journal.dynamo import ensure_table
        ensure_table("fsp-journal-test", region="us-east-1")
        # Second call must not raise
        ensure_table("fsp-journal-test", region="us-east-1")
        ddb = boto3.client("dynamodb", region_name="us-east-1")
        assert "fsp-journal-test" in ddb.list_tables()["TableNames"]


# ── Intraday signal lifecycle ────────────────────────────────────────────────

def test_log_intraday_and_retrieve(dynamo_journal):
    sig = _mk_signal()
    sid = dynamo_journal.log_intraday_signal(sig, "EURUSD|TREND_RSI|long", sent=True)
    assert sid.startswith("2026-06-17T08:00:00")

    unresolved = dynamo_journal.unresolved_signals("TREND_RSI")
    assert len(unresolved) == 1
    r = unresolved[0]
    assert r["pair"] == "EURUSD"
    assert r["entry"] == 1.1000  # Decimal -> float
    assert r["context"]["session"] == "LONDON"


def test_log_intraday_idempotent(dynamo_journal):
    """Same signal_id logged twice should not duplicate."""
    sig = _mk_signal()
    dk = "EURUSD|TREND_RSI|long"
    sid1 = dynamo_journal.log_intraday_signal(sig, dk, sent=True)
    sid2 = dynamo_journal.log_intraday_signal(sig, dk, sent=True)
    assert sid1 == sid2
    assert len(dynamo_journal.unresolved_signals("TREND_RSI")) == 1


def test_update_outcome_moves_to_resolved(dynamo_journal):
    sig = _mk_signal()
    sid = dynamo_journal.log_intraday_signal(sig, "EURUSD|TREND_RSI|long", sent=True)

    assert len(dynamo_journal.unresolved_signals("TREND_RSI")) == 1

    dynamo_journal.update_outcome(sid, "win", 3.0, "2026-06-17T10:00:00+00:00")

    # No longer unresolved
    assert len(dynamo_journal.unresolved_signals("TREND_RSI")) == 0
    # Now resolved
    resolved = dynamo_journal.resolved_signals("TREND_RSI")
    assert len(resolved) == 1
    assert resolved[0]["outcome"] == "win"
    assert resolved[0]["r_multiple"] == 3.0


def test_update_features(dynamo_journal):
    sig = _mk_signal()
    sid = dynamo_journal.log_intraday_signal(sig, "EURUSD|TREND_RSI|long", sent=False)
    dynamo_journal.update_features(sid, {"rsi": 28.5, "atr": 0.0012, "trend": "up"})
    item = dynamo_journal.table.get_item(Key={"signal_id": sid})["Item"]
    assert float(item["features"]["rsi"]) == 28.5
    assert item["features"]["trend"] == "up"


def test_update_meta_prediction(dynamo_journal):
    sig = _mk_signal()
    sid = dynamo_journal.log_intraday_signal(sig, "EURUSD|TREND_RSI|long", sent=False)
    dynamo_journal.update_meta_prediction(sid, 0.72)
    item = dynamo_journal.table.get_item(Key={"signal_id": sid})["Item"]
    assert float(item["meta_p_win"]) == 0.72


# ── Dedup lookup ─────────────────────────────────────────────────────────────

def test_last_signal_dedup_key_by_strategy(dynamo_journal):
    now = datetime.now(timezone.utc).isoformat()
    sig = _mk_signal(ts=now)
    dk = "EURUSD|TREND_RSI|long"
    dynamo_journal.log_intraday_signal(sig, dk, sent=True)

    found = dynamo_journal.last_signal_dedup_key("EURUSD", minutes=120,
                                                 strategy="TREND_RSI")
    assert found == dk


def test_last_signal_dedup_key_respects_window(dynamo_journal):
    """Old signal outside the window should not be returned."""
    old_ts = "2020-01-01T00:00:00+00:00"
    sig = _mk_signal(ts=old_ts)
    dynamo_journal.log_intraday_signal(sig, "EURUSD|TREND_RSI|long", sent=True)
    found = dynamo_journal.last_signal_dedup_key("EURUSD", minutes=60,
                                                 strategy="TREND_RSI")
    assert found is None


def test_last_signal_dedup_key_strategy_filter(dynamo_journal):
    """Querying a different strategy should not match."""
    now = datetime.now(timezone.utc).isoformat()
    dynamo_journal.log_intraday_signal(_mk_signal(ts=now, strategy="TREND_RSI"),
                                       "EURUSD|TREND_RSI|long", sent=True)
    found = dynamo_journal.last_signal_dedup_key("EURUSD", minutes=120,
                                                 strategy="ASIA_HL")
    assert found is None


# ── 4SP setup ────────────────────────────────────────────────────────────────

def test_log_4sp_setup(dynamo_journal):
    setup = FakeSetup(
        pair="GBPUSD", grade=FakeGrade("A+"), direction="short",
        entry=1.2700, sl=1.2750, tp1=1.2600, tp2=1.2550, risk_r=1.0,
        checklist=[FakeCheck("HTF", True, "aligned"), FakeCheck("OB", True)],
        context={"session": "NY_AM", "premium": True},
    )
    sid = dynamo_journal.log_signal(setup, "GBPUSD|A+|short|1.27|1.275", sent=True)
    item = dynamo_journal.table.get_item(Key={"signal_id": sid})["Item"]
    assert item["pair"] == "GBPUSD"
    assert item["grade"] == "A+"
    assert item["table_type"] == "4sp"
    assert item["context"]["premium"] is True


# ── NaN / inf handling ───────────────────────────────────────────────────────

def test_nan_values_dropped(dynamo_journal):
    """NaN/inf must not crash the DynamoDB write (gets converted to None/dropped)."""
    sig = _mk_signal()
    sig.tp2 = float("nan")
    sig.inv_pips = float("inf")
    sid = dynamo_journal.log_intraday_signal(sig, "EURUSD|TREND_RSI|long", sent=True)
    item = dynamo_journal.table.get_item(Key={"signal_id": sid})["Item"]
    # NaN/inf fields should be absent (converted to None then dropped)
    assert "tp2" not in item or item.get("tp2") is None
    assert "inv_pips" not in item or item.get("inv_pips") is None


# ── Backend dispatch (db.py routing) ─────────────────────────────────────────

def test_db_routes_to_dynamo_when_enabled():
    """db.py public functions should delegate to DynamoJournal when env set."""
    with mock_aws():
        os.environ["AWS_DEFAULT_REGION"] = "us-east-1"
        os.environ["FSP_JOURNAL_BACKEND"] = "dynamo"
        os.environ["FSP_DYNAMO_TABLE"] = "fsp-journal-test"
        try:
            from fsp.journal.dynamo import ensure_table
            ensure_table("fsp-journal-test", region="us-east-1")

            # Reset the cached backend so it picks up the test table
            import fsp.journal.db as db
            db._dynamo_backend = None

            sig = _mk_signal()
            sid = db.log_intraday_signal(sig, "EURUSD|TREND_RSI|long", sent=True)
            assert isinstance(sid, str)  # DynamoDB returns string id, not int

            unresolved = db.unresolved_signals("TREND_RSI")
            assert len(unresolved) == 1
        finally:
            os.environ.pop("FSP_JOURNAL_BACKEND", None)
            os.environ.pop("FSP_DYNAMO_TABLE", None)
            db._dynamo_backend = None
