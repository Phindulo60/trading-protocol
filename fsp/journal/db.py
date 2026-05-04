"""SQLite trade journal — logs every 4SP setup and intraday signal sent."""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

DB_PATH = Path.home() / ".fsp" / "journal.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    pair TEXT NOT NULL,
    grade TEXT NOT NULL,
    direction TEXT,
    entry REAL, sl REAL, tp1 REAL, tp2 REAL,
    risk_r REAL,
    checklist_json TEXT,
    context_json TEXT,
    sent_to_telegram INTEGER DEFAULT 0,
    dedup_key TEXT,
    outcome TEXT           -- 'win' | 'loss' | 'be' | null
);
CREATE INDEX IF NOT EXISTS ix_signals_dedup ON signals(dedup_key, ts);
CREATE INDEX IF NOT EXISTS ix_signals_pair_ts ON signals(pair, ts);

CREATE TABLE IF NOT EXISTS intraday_signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    pair TEXT NOT NULL,
    strategy TEXT NOT NULL,     -- 'ECM' | 'ARB'
    direction TEXT NOT NULL,
    entry REAL, sl REAL, tp1 REAL, tp2 REAL,
    inv_pips REAL,
    rr_tp1 REAL,
    risk_r REAL,
    note TEXT,
    dedup_key TEXT UNIQUE,
    sent_to_telegram INTEGER DEFAULT 0,
    context_json TEXT,
    outcome TEXT,              -- 'win' | 'loss' | 'timeout'
    r_multiple REAL,           -- actual R achieved (filled by resolver)
    exit_ts TEXT               -- when the trade ended (filled by resolver)
);
CREATE INDEX IF NOT EXISTS ix_intraday_pair_ts ON intraday_signals(pair, ts);
CREATE INDEX IF NOT EXISTS ix_intraday_dedup ON intraday_signals(dedup_key, ts);
"""


def _conn() -> sqlite3.Connection:
    """Return an open connection with schema applied. Caller must commit+close."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(DB_PATH)
    c.executescript(SCHEMA)
    return c


@contextmanager
def conn():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(DB_PATH)
    c.executescript(SCHEMA)
    try:
        yield c
        c.commit()
    finally:
        c.close()


def last_signal_dedup_key(pair: str, minutes: int = 60,
                          strategy: str | None = None) -> str | None:
    """Return the most recent dedup key for pair within the last `minutes`.

    If strategy is given, look in intraday_signals table for that strategy.
    Otherwise look in the 4SP signals table.
    """
    cutoff = (datetime.now(timezone.utc) - timedelta(minutes=minutes)).isoformat()
    with conn() as c:
        if strategy is not None:
            r = c.execute(
                "SELECT dedup_key FROM intraday_signals "
                "WHERE pair=? AND strategy=? AND ts>=? "
                "ORDER BY ts DESC LIMIT 1",
                (pair, strategy, cutoff)
            ).fetchone()
        else:
            r = c.execute(
                "SELECT dedup_key FROM signals WHERE pair=? AND ts>=? "
                "ORDER BY ts DESC LIMIT 1",
                (pair, cutoff)
            ).fetchone()
    return r[0] if r else None


def log_signal(setup, dedup_key: str, sent: bool) -> int:
    import json
    with conn() as c:
        cur = c.execute(
            "INSERT INTO signals (ts, pair, grade, direction, entry, sl, tp1, tp2, "
            "risk_r, checklist_json, context_json, sent_to_telegram, dedup_key) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (datetime.now(timezone.utc).isoformat(), setup.pair, setup.grade.value,
             setup.direction, setup.entry, setup.sl, setup.tp1, setup.tp2,
             setup.risk_r,
             json.dumps([(c_.name, bool(c_.passed), c_.note) for c_ in setup.checklist]),
             json.dumps({k: (bool(v) if isinstance(v, bool) else v)
                         for k, v in setup.context.items()}, default=str),
             1 if sent else 0, dedup_key),
        )
        return cur.lastrowid


def log_intraday_signal(sig, dedup_key: str, sent: bool) -> int:
    import json
    with conn() as c:
        cur = c.execute(
            "INSERT OR IGNORE INTO intraday_signals "
            "(ts, pair, strategy, direction, entry, sl, tp1, tp2, "
            "inv_pips, rr_tp1, risk_r, note, dedup_key, sent_to_telegram, context_json) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (sig.ts, sig.pair, sig.strategy, sig.direction,
             sig.entry, sig.sl, sig.tp1, sig.tp2,
             sig.inv_pips, sig.rr_tp1, sig.risk_r,
             sig.note, dedup_key, int(sent),
             json.dumps(sig.context)),
        )
        return cur.lastrowid


def migrate(c: sqlite3.Connection) -> None:
    """Add columns introduced after initial schema (safe to re-run)."""
    for col, typedef in [
        ("r_multiple", "REAL"),
        ("exit_ts",    "TEXT"),
    ]:
        try:
            c.execute(f"ALTER TABLE intraday_signals ADD COLUMN {col} {typedef}")
            c.commit()
        except sqlite3.OperationalError:
            pass  # column already exists


def update_outcome(signal_id: int, outcome: str,
                   r_multiple: float, exit_ts: str) -> None:
    """Write resolved outcome back to a logged signal."""
    with conn() as c:
        migrate(c)
        c.execute(
            "UPDATE intraday_signals SET outcome=?, r_multiple=?, exit_ts=? WHERE id=?",
            (outcome, r_multiple, exit_ts, signal_id),
        )


def unresolved_signals(strategy: str = "TREND_RSI") -> list[dict]:
    """Return all signals without an outcome yet."""
    with conn() as c:
        migrate(c)
        rows = c.execute(
            "SELECT id, ts, pair, direction, entry, sl, tp1, rr_tp1, context_json "
            "FROM intraday_signals "
            "WHERE outcome IS NULL AND strategy=? "
            "ORDER BY ts ASC",
            (strategy,),
        ).fetchall()
    return [
        {"id": r[0], "ts": r[1], "pair": r[2], "direction": r[3],
         "entry": r[4], "sl": r[5], "tp1": r[6], "rr_tp1": r[7],
         "context": __import__("json").loads(r[8] or "{}")}
        for r in rows
    ]


def resolved_signals(strategy: str = "TREND_RSI") -> list[dict]:
    """Return all signals with outcomes filled — used by the review command."""
    with conn() as c:
        migrate(c)
        rows = c.execute(
            "SELECT id, ts, pair, direction, entry, sl, tp1, rr_tp1, "
            "outcome, r_multiple, exit_ts, context_json "
            "FROM intraday_signals "
            "WHERE outcome IS NOT NULL AND strategy=? "
            "ORDER BY ts ASC",
            (strategy,),
        ).fetchall()
    return [
        {"id": r[0], "ts": r[1], "pair": r[2], "direction": r[3],
         "entry": r[4], "sl": r[5], "tp1": r[6], "rr_tp1": r[7],
         "outcome": r[8], "r_multiple": r[9], "exit_ts": r[10],
         "context": __import__("json").loads(r[11] or "{}")}
        for r in rows
    ]
