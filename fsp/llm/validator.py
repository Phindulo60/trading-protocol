"""LLM Trading Analyst — uses Bedrock Claude Sonnet as a decision-making co-pilot.

Not a filter. A trading analyst that receives full market context and makes
reasoned decisions about whether to take, skip, or modify a trade.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal

import boto3

log = logging.getLogger(__name__)

Decision = Literal["TAKE", "SKIP", "REDUCE"]

# Sonnet for deep reasoning; Haiku as fallback
DEFAULT_MODEL = "us.anthropic.claude-sonnet-4-20250514-v1:0"
FALLBACK_MODEL = "anthropic.claude-3-haiku-20240307-v1:0"


@dataclass
class ValidationResult:
    decision: Decision
    confidence: float       # 0.0 – 1.0
    reason: str             # one-line for logs
    analysis: str           # full reasoning (included in Telegram for TAKE/REDUCE)
    model_used: str
    suggested_tp: float | None = None   # LLM can suggest modified TP
    suggested_sl: float | None = None   # LLM can suggest modified SL


SYSTEM_PROMPT = """You are a professional forex trading analyst working alongside an algorithmic signal engine. You are NOT a filter — you are a co-pilot who understands market dynamics deeply.

## Your Role
When the algo fires a signal, you receive:
- The signal details (pair, direction, entry, SL, TP, strategy, R:R)
- Current trading session and day of week
- Upcoming economic events (next 4 hours)
- Recent trade history for this strategy + pair
- Current strategy performance stats (win rate, streak, net R)
- Other signals fired today (correlation/exposure check)
- Technical context (M15/H1 trend, ATR, price)

## What You Do
Analyse ALL the context and make a decisive trading call. Think like a senior trader reviewing a junior's trade idea.

Consider:
1. **Economic calendar** — Is there a high-impact event that could blow through stops? Events within 30 min = almost certainly skip. Within 2h = high caution.
2. **Session quality** — EUR/GBP pairs in Asia session = poor liquidity = wider spreads = worse fills. JPY in Asia = fine. The LO-NY overlap (12-16 UTC) is premium time for all pairs.
3. **Strategy momentum** — Is this strategy in a winning or losing streak? A strategy on a 5-loss streak might be in a regime change. But also don't abandon a strategy just because of 2 losses.
4. **Correlation exposure** — If we already have EURUSD long and now GBPUSD long fires, that's doubled USD-short exposure. Flag it.
5. **Technical alignment** — Does the M15 signal align with H1 trend? Counter-trend entries need higher conviction.
6. **Day of week** — Monday = range establishment, lower conviction. Friday = position squaring, erratic moves after 16 UTC.
7. **R:R quality** — Is the risk:reward justified given the context?

## Decision Framework
- **TAKE**: Context supports the trade. Include a brief conviction note for the trader.
- **SKIP**: Something material argues against it (event risk, correlation overload, terrible session, regime shift). Be specific about WHY.
- **REDUCE**: Trade has merit but risk is elevated. Specify why half-size is prudent.

## Response Format
Return EXACTLY this JSON:
{
  "decision": "TAKE" | "SKIP" | "REDUCE",
  "confidence": 0.0-1.0,
  "reason": "one-line summary for logs",
  "analysis": "2-4 sentence analysis with your actual reasoning. Be specific. Reference the data you were given.",
  "suggested_tp": null,
  "suggested_sl": null
}

You may suggest modified TP/SL values if you think the algo's levels are suboptimal given the context (e.g., TP sitting right at a major event time, SL too tight for current ATR). Set to null if no modification needed.

Be decisive. Don't hedge with "it could go either way." Make a call."""


class SignalValidator:
    """LLM trading analyst powered by Bedrock."""

    def __init__(self, region: str = "us-east-1", model_id: str = DEFAULT_MODEL):
        self._client = boto3.client("bedrock-runtime", region_name=region)
        self._model_id = model_id
        self._fallback_id = FALLBACK_MODEL

    def validate(
        self,
        pair: str,
        direction: str,
        strategy: str,
        entry: float,
        sl: float,
        tp: float,
        context: dict | None = None,
    ) -> ValidationResult:
        """Full-context LLM analysis of a trading signal."""
        ctx = context or {}
        user_msg = self._build_prompt(pair, direction, strategy, entry, sl, tp, ctx)

        # Try Sonnet first, fall back to Haiku on failure
        for model_id in [self._model_id, self._fallback_id]:
            try:
                return self._call_bedrock(user_msg, model_id)
            except Exception as e:
                log.warning("LLM call failed (%s) with %s: %s",
                           type(e).__name__, model_id, e)
                if model_id == self._fallback_id:
                    # Both failed — default to TAKE
                    return ValidationResult(
                        decision="TAKE",
                        confidence=0.5,
                        reason=f"LLM unavailable ({type(e).__name__}), defaulting to TAKE",
                        analysis="Could not reach any LLM model. Passing signal through unvalidated.",
                        model_used="fallback",
                    )

    def _build_prompt(self, pair: str, direction: str, strategy: str,
                      entry: float, sl: float, tp: float, ctx: dict) -> str:
        risk = abs(entry - sl)
        rr = abs(tp - entry) / risk if risk > 0 else 0

        lines = [
            "# Signal for Review",
            f"**{pair} {direction.upper()}** via {strategy}",
            f"- Entry: {entry:.5f}",
            f"- SL: {sl:.5f} ({abs(entry - sl) / entry * 100:.3f}%)",
            f"- TP: {tp:.5f} (R:R = {rr:.1f})",
            f"- Signal time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        ]

        # Session
        session = ctx.get("session", {})
        if session:
            lines.append(f"\n## Trading Session")
            lines.append(f"- Current: **{session.get('name', '?')}** — {session.get('time', '?')}")
            if session.get("note"):
                lines.append(f"- Note: {session['note']}")

        # Economic calendar
        events = ctx.get("upcoming_events", [])
        if events:
            lines.append(f"\n## Economic Calendar (next 4h)")
            for ev in events:
                mins = ev.get("minutes_away", 0)
                urgency = "⚠️" if abs(mins) < 60 else ""
                direction_str = "ago" if mins < 0 else "away"
                forecast_str = ""
                if ev.get("forecast"):
                    forecast_str = f" | Forecast: {ev['forecast']}  Prev: {ev['previous']}"
                lines.append(
                    f"- {urgency}[{ev['impact']}] **{ev['event']}** ({ev['currency']}) "
                    f"at {ev['time']} — {abs(mins)} min {direction_str}{forecast_str}"
                )
        else:
            lines.append("\n## Economic Calendar\nNo high/medium impact events in the next 4 hours.")

        # Strategy performance
        stats = ctx.get("strategy_stats", {})
        if stats.get("total", 0) > 0:
            lines.append(f"\n## {strategy} Performance (last {stats['total']} trades)")
            lines.append(f"- Win rate: {stats['win_rate']}%")
            lines.append(f"- Current streak: {stats['streak']}")
            lines.append(f"- Net R: {stats['net_r']:+.1f}")

        # Recent trades on this pair
        recent = ctx.get("recent_trades", [])
        if recent:
            lines.append(f"\n## Recent {strategy} Trades on {pair}")
            for t in recent[:6]:
                icon = "W" if t["outcome"] == "win" else ("L" if t["outcome"] == "loss" else "T")
                lines.append(f"- [{icon}] {t['direction']} {t.get('r', 0):+.1f}R ({t['ts'][:10]})")

        # Other signals today (exposure check)
        today_sigs = ctx.get("signals_today", [])
        if today_sigs:
            lines.append(f"\n## Other Signals Fired Today")
            for s in today_sigs:
                lines.append(f"- {s['pair']} {s.get('direction', '?')} ({s.get('strategy', '?')}) — {s.get('outcome', 'open')}")

        # Price / technical context
        price = ctx.get("price", {})
        if price:
            lines.append(f"\n## Technical Context")
            if "last_price" in price:
                lines.append(f"- Last price: {price['last_price']}")
            if "m15_trend" in price:
                lines.append(f"- M15 trend: {price['m15_trend']}")
            if "h1_trend" in price:
                lines.append(f"- H1 trend: {price['h1_trend']}")
            if "m15_atr" in price:
                lines.append(f"- M15 ATR(14): {price['m15_atr']}")
            if "today_range" in price:
                lines.append(f"- Today's range: {price['today_range']}")

        return "\n".join(lines)

    def _call_bedrock(self, user_msg: str, model_id: str) -> ValidationResult:
        """Call Bedrock Converse API."""
        response = self._client.converse(
            modelId=model_id,
            messages=[{"role": "user", "content": [{"text": user_msg}]}],
            system=[{"text": SYSTEM_PROMPT}],
            inferenceConfig={"maxTokens": 500, "temperature": 0.2},
        )

        output_text = response["output"]["message"]["content"][0]["text"]

        try:
            text = output_text.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            parsed = json.loads(text)
        except json.JSONDecodeError:
            log.warning("LLM returned non-JSON: %s", output_text[:200])
            return ValidationResult(
                decision="TAKE", confidence=0.5,
                reason=f"Parse error, raw: {output_text[:100]}",
                analysis=output_text[:300],
                model_used=model_id,
            )

        decision = parsed.get("decision", "TAKE").upper()
        if decision not in ("TAKE", "SKIP", "REDUCE"):
            decision = "TAKE"

        return ValidationResult(
            decision=decision,
            confidence=float(parsed.get("confidence", 0.7)),
            reason=parsed.get("reason", "no reason given"),
            analysis=parsed.get("analysis", parsed.get("reason", "")),
            model_used=model_id,
            suggested_tp=parsed.get("suggested_tp"),
            suggested_sl=parsed.get("suggested_sl"),
        )
