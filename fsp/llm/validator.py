"""LLM Signal Validator — uses Bedrock to assess signal quality before notification.

Adds a contextual layer: economic events, correlation, recent performance.
Returns TAKE / SKIP / REDUCE with reasoning.
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

# Use Claude Sonnet for quality; Nova Lite for cost savings
DEFAULT_MODEL = "anthropic.claude-3-haiku-20240307-v1:0"  # Fast + cheap for validation
FALLBACK_MODEL = "anthropic.claude-3-5-haiku-20241022-v1:0"


@dataclass
class ValidationResult:
    decision: Decision
    confidence: float  # 0-1
    reason: str
    model_used: str


SYSTEM_PROMPT = """You are a forex trading signal validator. Your job is to assess whether a signal should be TAKEN, SKIPPED, or REDUCED (half position size).

Consider:
1. Market context (trend alignment, session timing, volatility)
2. Economic calendar (high-impact events within 2h = dangerous)
3. Correlation risk (multiple positions in same direction on correlated pairs)
4. Recent signal performance for this strategy/pair
5. Time of day (London/NY overlap = best liquidity; Asian = lower vol for majors)

Rules:
- If a high-impact event (NFP, FOMC, CPI, ECB) is within 2 hours: SKIP
- If 2+ open signals in same directional exposure: REDUCE
- If strategy has lost 3+ consecutive trades on this pair recently: SKIP
- If signal aligns with higher timeframe trend AND good session timing: TAKE
- Default to TAKE if no clear reason to skip

Respond with EXACTLY this JSON format:
{"decision": "TAKE|SKIP|REDUCE", "confidence": 0.0-1.0, "reason": "one-line explanation"}"""


class SignalValidator:
    """Validates trading signals using Bedrock LLM."""

    def __init__(self, region: str = "us-east-1", model_id: str = DEFAULT_MODEL):
        self._client = boto3.client("bedrock-runtime", region_name=region)
        self._model_id = model_id

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
        """Validate a signal with optional market context.

        Args:
            context: Optional dict with keys like:
                - session: current session (ASIA/LO/NY-AM/NY-PM)
                - h4_trend: bull/bear/neutral
                - dxy_bias: bull/bear/flat
                - upcoming_events: list of {"event": str, "time": str, "impact": str}
                - recent_trades: list of {"pair": str, "result": str, "r": float}
                - open_positions: list of {"pair": str, "direction": str}
        """
        ctx = context or {}

        user_msg = self._build_prompt(pair, direction, strategy, entry, sl, tp, ctx)

        try:
            result = self._call_bedrock(user_msg)
            return result
        except Exception as e:
            log.warning("LLM validation failed (%s), defaulting to TAKE: %s", type(e).__name__, e)
            return ValidationResult(
                decision="TAKE",
                confidence=0.5,
                reason=f"LLM unavailable ({type(e).__name__}), defaulting to TAKE",
                model_used="fallback",
            )

    def _build_prompt(self, pair: str, direction: str, strategy: str,
                      entry: float, sl: float, tp: float, ctx: dict) -> str:
        lines = [
            f"## Signal",
            f"- Pair: {pair}",
            f"- Direction: {direction}",
            f"- Strategy: {strategy}",
            f"- Entry: {entry:.5f}",
            f"- SL: {sl:.5f} ({abs(entry - sl) / entry * 100:.2f}%)",
            f"- TP: {tp:.5f} (R:R = {abs(tp - entry) / abs(entry - sl):.1f})",
            f"- Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        ]

        if ctx.get("session"):
            lines.append(f"\n## Session: {ctx['session']}")

        if ctx.get("h4_trend"):
            lines.append(f"## H4 Trend: {ctx['h4_trend']}")

        if ctx.get("dxy_bias"):
            lines.append(f"## DXY Bias: {ctx['dxy_bias']}")

        if ctx.get("upcoming_events"):
            lines.append("\n## Upcoming Economic Events (next 4h)")
            for ev in ctx["upcoming_events"]:
                lines.append(f"- [{ev.get('impact', '?')}] {ev['event']} at {ev.get('time', '?')}")

        if ctx.get("recent_trades"):
            lines.append(f"\n## Recent {strategy} trades on {pair}")
            for t in ctx["recent_trades"][-5:]:
                lines.append(f"- {t.get('result', '?')} ({t.get('r', 0):+.1f}R)")

        if ctx.get("open_positions"):
            lines.append("\n## Currently Open Positions")
            for p in ctx["open_positions"]:
                lines.append(f"- {p['pair']} {p['direction']}")

        return "\n".join(lines)

    def _call_bedrock(self, user_msg: str) -> ValidationResult:
        """Call Bedrock Converse API."""
        response = self._client.converse(
            modelId=self._model_id,
            messages=[{"role": "user", "content": [{"text": user_msg}]}],
            system=[{"text": SYSTEM_PROMPT}],
            inferenceConfig={"maxTokens": 150, "temperature": 0.1},
        )

        output_text = response["output"]["message"]["content"][0]["text"]

        # Parse JSON response
        try:
            # Handle potential markdown wrapping
            text = output_text.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0]
            parsed = json.loads(text)
        except json.JSONDecodeError:
            log.warning("LLM returned non-JSON: %s", output_text[:200])
            return ValidationResult(
                decision="TAKE", confidence=0.5,
                reason=f"Parse error, raw: {output_text[:100]}",
                model_used=self._model_id,
            )

        decision = parsed.get("decision", "TAKE").upper()
        if decision not in ("TAKE", "SKIP", "REDUCE"):
            decision = "TAKE"

        return ValidationResult(
            decision=decision,
            confidence=float(parsed.get("confidence", 0.7)),
            reason=parsed.get("reason", "no reason given"),
            model_used=self._model_id,
        )
