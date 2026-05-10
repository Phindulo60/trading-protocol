"""LLM Backtest — replay historical signals through the analyst and measure impact.

Reconstructs the context available at each signal's timestamp, asks the LLM
for its decision, then compares against actual outcomes.

Usage:
    fsp llm-backtest              # run full backtest
    fsp llm-backtest --limit 20   # quick test with 20 signals
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fsp.config import data_dir

log = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    signal_id: int
    pair: str
    strategy: str
    direction: str
    entry: float
    actual_outcome: str       # win/loss/timeout
    actual_r: float
    llm_decision: str         # TAKE/SKIP/REDUCE
    llm_confidence: float
    llm_reason: str
    llm_analysis: str


@dataclass
class BacktestSummary:
    results: list[BacktestResult] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.results)

    def _filtered(self, actual: str | None = None, decision: str | None = None):
        r = self.results
        if actual:
            r = [x for x in r if x.actual_outcome == actual]
        if decision:
            r = [x for x in r if x.llm_decision == decision]
        return r

    @property
    def without_llm(self) -> dict:
        """Baseline: take every signal."""
        r = self.results
        net = sum(x.actual_r for x in r)
        wins = sum(1 for x in r if x.actual_outcome == "win")
        return {"trades": len(r), "wins": wins, "wr": wins/len(r)*100 if r else 0, "net_r": round(net, 1)}

    @property
    def with_llm(self) -> dict:
        """With LLM: skip SKIP, half-size REDUCE."""
        taken = [x for x in self.results if x.llm_decision == "TAKE"]
        reduced = [x for x in self.results if x.llm_decision == "REDUCE"]
        skipped = [x for x in self.results if x.llm_decision == "SKIP"]

        net = sum(x.actual_r for x in taken) + sum(x.actual_r * 0.5 for x in reduced)
        trades = len(taken) + len(reduced)
        wins = sum(1 for x in taken if x.actual_outcome == "win") + \
               sum(1 for x in reduced if x.actual_outcome == "win")
        return {
            "trades": trades, "skipped": len(skipped),
            "wins": wins, "wr": wins/trades*100 if trades else 0,
            "net_r": round(net, 1),
        }

    @property
    def skip_accuracy(self) -> dict:
        """How good is the LLM at skipping losers?"""
        skipped = self._filtered(decision="SKIP")
        if not skipped:
            return {"total_skipped": 0, "were_losers": 0, "accuracy": 0}
        losers = sum(1 for x in skipped if x.actual_outcome == "loss")
        return {
            "total_skipped": len(skipped),
            "were_losers": losers,
            "were_winners": sum(1 for x in skipped if x.actual_outcome == "win"),
            "accuracy": round(losers / len(skipped) * 100, 1),
        }

    def report(self) -> str:
        """Generate human-readable backtest report."""
        baseline = self.without_llm
        llm = self.with_llm
        skip = self.skip_accuracy

        lines = [
            "=" * 60,
            "LLM ANALYST BACKTEST REPORT",
            "=" * 60,
            "",
            f"Signals tested: {self.total}",
            "",
            "── Baseline (take everything) ──",
            f"  Trades: {baseline['trades']}",
            f"  Wins: {baseline['wins']} ({baseline['wr']:.1f}%)",
            f"  Net R: {baseline['net_r']:+.1f}",
            "",
            "── With Opus Analyst ──",
            f"  Trades taken: {llm['trades']}  (skipped {llm['skipped']})",
            f"  Wins: {llm['wins']} ({llm['wr']:.1f}%)",
            f"  Net R: {llm['net_r']:+.1f}",
            f"  R improvement: {llm['net_r'] - baseline['net_r']:+.1f}",
            "",
            "── Skip Accuracy ──",
            f"  Total skipped: {skip['total_skipped']}",
            f"  Were actually losers: {skip['were_losers']} ({skip['accuracy']:.1f}%)",
            f"  Were actually winners: {skip.get('were_winners', 0)}",
            "",
        ]

        # Per-strategy breakdown
        strategies = set(r.strategy for r in self.results)
        for strat in sorted(strategies):
            strat_results = [r for r in self.results if r.strategy == strat]
            taken = [r for r in strat_results if r.llm_decision != "SKIP"]
            skipped = [r for r in strat_results if r.llm_decision == "SKIP"]
            base_r = sum(r.actual_r for r in strat_results)
            llm_r = sum(r.actual_r * (0.5 if r.llm_decision == "REDUCE" else 1.0)
                       for r in taken)
            lines.append(f"── {strat} ──")
            lines.append(f"  Baseline: {len(strat_results)} trades, {base_r:+.1f}R")
            lines.append(f"  With LLM: {len(taken)} trades, {llm_r:+.1f}R  (skipped {len(skipped)})")
            lines.append("")

        lines.append("=" * 60)
        return "\n".join(lines)


def _reconstruct_context(signal: dict, all_signals: list[dict]) -> dict:
    """Reconstruct the context that was available at signal time."""
    from fsp.llm.context import _SESSION_NOTES

    sig_ts = datetime.fromisoformat(signal["ts"])
    if sig_ts.tzinfo is None:
        sig_ts = sig_ts.replace(tzinfo=timezone.utc)

    # Session at signal time
    hour = sig_ts.hour
    sessions = [("ASIA", 0, 8), ("LO", 8, 12), ("NY-AM", 12, 16),
                ("NY-PM", 16, 21), ("OFF", 21, 24)]
    session_name = "OFF"
    for name, start, end in sessions:
        if start <= hour < end:
            session_name = name
            break

    session = {
        "name": session_name,
        "time": sig_ts.strftime("%H:%M UTC (%A)"),
        "note": _SESSION_NOTES.get(session_name, ""),
    }

    # Recent trades BEFORE this signal
    prior = [s for s in all_signals
             if s["ts"] < signal["ts"] and s.get("outcome")]
    pair_prior = [s for s in prior if s["pair"] == signal["pair"]
                  and s["strategy"] == signal["strategy"]][-8:]
    recent_trades = [
        {"pair": s["pair"], "strategy": s["strategy"], "direction": s["direction"],
         "outcome": s["outcome"], "r": s.get("r_multiple", 0), "ts": s["ts"]}
        for s in pair_prior
    ]

    # Strategy stats from prior signals
    strat_prior = [s for s in prior if s["strategy"] == signal["strategy"]][-20:]
    total = len(strat_prior)
    wins = sum(1 for s in strat_prior if s["outcome"] == "win")
    net_r = sum(s.get("r_multiple", 0) or 0 for s in strat_prior)
    streak_type = strat_prior[-1]["outcome"] if strat_prior else "none"
    streak_count = 0
    for s in reversed(strat_prior):
        if s["outcome"] == streak_type:
            streak_count += 1
        else:
            break

    strategy_stats = {
        "total": total,
        "win_rate": round(wins / total * 100, 1) if total else 0,
        "streak": f"{streak_count} {streak_type}{'s' if streak_count > 1 else ''}",
        "net_r": round(net_r, 1),
    }

    # Other signals same day (for exposure check)
    day_start = sig_ts.replace(hour=0, minute=0, second=0).isoformat()
    signals_today = [
        {"pair": s["pair"], "direction": s["direction"],
         "strategy": s["strategy"], "outcome": s.get("outcome", "open")}
        for s in all_signals
        if s["ts"][:10] == signal["ts"][:10]
        and s["ts"] < signal["ts"]
        and s["pair"] != signal["pair"]
    ]

    return {
        "session": session,
        "upcoming_events": [],  # can't reconstruct historical calendar
        "recent_trades": recent_trades,
        "strategy_stats": strategy_stats,
        "signals_today": signals_today,
        "price": {},  # would need historical data to reconstruct
    }


def run_backtest(limit: int | None = None, strategy: str | None = None,
                 verbose: bool = True) -> BacktestSummary:
    """Replay historical signals through the LLM analyst."""
    from fsp.journal.db import conn, migrate
    from fsp.llm.validator import SignalValidator

    # Load all resolved signals
    with conn() as c:
        migrate(c)
        query = ("SELECT id, ts, pair, strategy, direction, entry, sl, tp1, rr_tp1, "
                 "outcome, r_multiple, context_json "
                 "FROM intraday_signals WHERE outcome IS NOT NULL ")
        params: list = []
        if strategy:
            query += "AND strategy=? "
            params.append(strategy)
        query += "ORDER BY ts ASC"
        rows = c.execute(query, params).fetchall()

    all_signals = [
        {"id": r[0], "ts": r[1], "pair": r[2], "strategy": r[3], "direction": r[4],
         "entry": r[5], "sl": r[6], "tp1": r[7], "rr_tp1": r[8],
         "outcome": r[9], "r_multiple": r[10],
         "context": json.loads(r[11] or "{}")}
        for r in rows
    ]

    if limit:
        all_signals = all_signals[:limit]

    if verbose:
        print(f"Backtesting {len(all_signals)} signals through LLM analyst...")

    validator = SignalValidator()
    summary = BacktestSummary()

    for i, sig in enumerate(all_signals):
        ctx = _reconstruct_context(sig, all_signals)

        try:
            result = validator.validate(
                pair=sig["pair"],
                direction=sig["direction"],
                strategy=sig["strategy"],
                entry=sig["entry"],
                sl=sig["sl"],
                tp=sig["tp1"],
                context=ctx,
            )

            br = BacktestResult(
                signal_id=sig["id"],
                pair=sig["pair"],
                strategy=sig["strategy"],
                direction=sig["direction"],
                entry=sig["entry"],
                actual_outcome=sig["outcome"],
                actual_r=sig.get("r_multiple", 0) or 0,
                llm_decision=result.decision,
                llm_confidence=result.confidence,
                llm_reason=result.reason,
                llm_analysis=result.analysis,
            )
            summary.results.append(br)

            if verbose:
                icon = {"win": "W", "loss": "L", "timeout": "T"}[sig["outcome"]]
                llm_icon = {"TAKE": ">>", "SKIP": "XX", "REDUCE": "//"}[result.decision]
                print(f"  [{i+1}/{len(all_signals)}] {sig['pair']} {sig['strategy']} "
                      f"{sig['direction']} → {icon} {sig.get('r_multiple', 0):+.1f}R "
                      f"| LLM: {llm_icon} {result.decision} ({result.confidence:.0%}) "
                      f"— {result.reason[:60]}")

        except Exception as e:
            log.error("Backtest failed for signal %d: %s", sig["id"], e)
            if verbose:
                print(f"  [{i+1}] ERROR: {e}")

        # Rate limiting courtesy — don't hammer Bedrock
        time.sleep(0.5)

    # Save results
    report = summary.report()
    report_path = data_dir() / "llm_backtest_report.txt"
    report_path.write_text(report)

    # Save raw results as JSON for learning
    raw_path = data_dir() / "llm_backtest_results.json"
    raw_data = [
        {"signal_id": r.signal_id, "pair": r.pair, "strategy": r.strategy,
         "direction": r.direction, "actual_outcome": r.actual_outcome,
         "actual_r": r.actual_r, "llm_decision": r.llm_decision,
         "llm_confidence": r.llm_confidence, "llm_reason": r.llm_reason,
         "llm_analysis": r.llm_analysis}
        for r in summary.results
    ]
    raw_path.write_text(json.dumps(raw_data, indent=2))

    if verbose:
        print(f"\n{report}")
        print(f"\nSaved to: {report_path}")

    return summary
