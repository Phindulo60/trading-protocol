"""LLM Learning Loop — analyses backtest results and generates a trading playbook.

After backtesting, the LLM reviews its own decisions vs outcomes to identify:
- What patterns it correctly caught (good skips)
- What it missed (bad takes / missed winners)
- Rules to add/remove from its decision framework

The playbook is stored as a text file and injected into the system prompt
for all future decisions — this is how the model "learns."
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import boto3

from fsp.config import data_dir

log = logging.getLogger(__name__)

PLAYBOOK_PATH = data_dir() / "trading_playbook.md"

REVIEW_PROMPT = """You are reviewing your own trading decisions from a backtest. You made calls on real signals, and now you can see what actually happened.

Your job: identify patterns in your HITS and MISSES, and write concrete rules for yourself to follow in future trading decisions.

## Format

Write a trading playbook in Markdown with these sections:

### Patterns I Caught (good decisions)
- Specific patterns where your SKIP saved money or TAKE made money

### Patterns I Missed (bad decisions)
- Signals you said TAKE that lost, or SKIP that would have won
- What you should have looked for

### Updated Decision Rules
- Concrete, actionable rules based on the evidence
- Be specific: "Skip TREND_RSI on EURUSD during NY-PM session" not "be more careful"

### Strategy Notes
- Per-strategy observations (which strategies to trust, which to be cautious with)

### Confidence Calibration
- Were your high-confidence calls more accurate than low-confidence ones?
- Should you adjust your confidence thresholds?

Be brutally honest about your mistakes. The goal is to get better, not to feel good."""


def generate_playbook(backtest_results: list[dict] | None = None,
                      verbose: bool = True) -> str:
    """Feed backtest results to the LLM and generate a trading playbook."""
    if backtest_results is None:
        results_path = data_dir() / "llm_backtest_results.json"
        if not results_path.exists():
            raise FileNotFoundError(
                "No backtest results found. Run `fsp llm-backtest` first.")
        backtest_results = json.loads(results_path.read_text())

    # Build the review prompt with all results
    lines = ["# Backtest Results for Review\n"]

    # Summary stats
    total = len(backtest_results)
    takes = [r for r in backtest_results if r["llm_decision"] == "TAKE"]
    skips = [r for r in backtest_results if r["llm_decision"] == "SKIP"]
    reduces = [r for r in backtest_results if r["llm_decision"] == "REDUCE"]

    take_wins = sum(1 for r in takes if r["actual_outcome"] == "win")
    skip_losers = sum(1 for r in skips if r["actual_outcome"] == "loss")
    skip_winners = sum(1 for r in skips if r["actual_outcome"] == "win")

    baseline_r = sum(r["actual_r"] for r in backtest_results)
    llm_r = (sum(r["actual_r"] for r in takes) +
             sum(r["actual_r"] * 0.5 for r in reduces))

    lines.append(f"Total signals: {total}")
    lines.append(f"You said TAKE: {len(takes)} (won {take_wins})")
    lines.append(f"You said SKIP: {len(skips)} (were losers: {skip_losers}, were winners: {skip_winners})")
    lines.append(f"You said REDUCE: {len(reduces)}")
    lines.append(f"Baseline net R (take all): {baseline_r:+.1f}")
    lines.append(f"Your net R: {llm_r:+.1f}")
    lines.append(f"R improvement: {llm_r - baseline_r:+.1f}")
    lines.append("")

    # Per-strategy breakdown
    strategies = sorted(set(r["strategy"] for r in backtest_results))
    for strat in strategies:
        sr = [r for r in backtest_results if r["strategy"] == strat]
        lines.append(f"\n## {strat} ({len(sr)} signals)")
        for r in sr:
            icon = {"win": "W", "loss": "L", "timeout": "T"}[r["actual_outcome"]]
            lines.append(
                f"- [{icon}] {r['pair']} {r['direction']} {r['actual_r']:+.1f}R "
                f"| You: {r['llm_decision']} ({r['llm_confidence']:.0%}) "
                f"— {r['llm_reason'][:80]}"
            )

    # Detailed misses (you said TAKE but it lost, or SKIP but it won)
    bad_takes = [r for r in takes if r["actual_outcome"] == "loss"]
    missed_wins = [r for r in skips if r["actual_outcome"] == "win"]

    if bad_takes:
        lines.append("\n\n## YOUR WORST CALLS (TAKE that lost)")
        for r in sorted(bad_takes, key=lambda x: x["actual_r"])[:10]:
            lines.append(f"- {r['pair']} {r['strategy']} {r['direction']} → {r['actual_r']:+.1f}R")
            lines.append(f"  Your reasoning: {r['llm_analysis'][:150]}")

    if missed_wins:
        lines.append("\n\n## MISSED WINNERS (SKIP that would have won)")
        for r in sorted(missed_wins, key=lambda x: -x["actual_r"])[:10]:
            lines.append(f"- {r['pair']} {r['strategy']} {r['direction']} → {r['actual_r']:+.1f}R")
            lines.append(f"  Your reasoning: {r['llm_analysis'][:150]}")

    user_msg = "\n".join(lines)

    if verbose:
        print("Sending backtest results to Opus for review...")

    # Call Opus to generate playbook
    client = boto3.client("bedrock-runtime", region_name="us-east-1")
    response = client.converse(
        modelId="us.anthropic.claude-opus-4-6-v1",
        messages=[{"role": "user", "content": [{"text": user_msg}]}],
        system=[{"text": REVIEW_PROMPT}],
        inferenceConfig={"maxTokens": 2000, "temperature": 0.3},
    )

    playbook = response["output"]["message"]["content"][0]["text"]

    # Add metadata header
    header = (
        f"# FSP Trading Playbook\n"
        f"_Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}_\n"
        f"_Based on: {total} signals backtested_\n"
        f"_Baseline: {baseline_r:+.1f}R → With LLM: {llm_r:+.1f}R "
        f"({llm_r - baseline_r:+.1f}R improvement)_\n\n"
    )
    full_playbook = header + playbook

    # Save playbook
    PLAYBOOK_PATH.write_text(full_playbook)

    if verbose:
        print(f"\n{full_playbook}")
        print(f"\nSaved to: {PLAYBOOK_PATH}")

    return full_playbook


def load_playbook() -> str | None:
    """Load the trading playbook for injection into the system prompt."""
    if PLAYBOOK_PATH.exists():
        return PLAYBOOK_PATH.read_text()
    return None
