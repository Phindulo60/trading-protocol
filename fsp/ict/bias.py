"""ICT higher-timeframe directional bias.

Bias is driven by market structure — the prevailing trend plus any recent,
high-conviction Market-Structure Shift. The dealing-range position is attached
as *context* (where to enter), not as a directional vote: in ICT you take the
HTF bias and then enter from the correct premium/discount half.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

from fsp.ict.structure import analyze_structure
from fsp.ict.premium_discount import dealing_range


@dataclass
class Bias:
    direction: Literal["bull", "bear", "neutral"]
    score: int                       # signed conviction (+ bull / - bear)
    reasons: list[str] = field(default_factory=list)
    range_position: float | None = None   # 0=low, 1=high within HTF dealing range
    range_zone: str | None = None          # premium / discount / equilibrium


def htf_bias(
    df: pd.DataFrame,
    swing_length: int = 5,
    atr_mult: float = 1.5,
    atr_len: int = 20,
) -> Bias:
    """Directional bias for a (higher-timeframe) frame."""
    st = analyze_structure(df, swing_length, atr_mult, atr_len)
    score = 0
    reasons: list[str] = []

    if st.trend == 1:
        score += 2
        reasons.append("HTF structure bullish (last break up)")
    elif st.trend == -1:
        score -= 2
        reasons.append("HTF structure bearish (last break down)")
    else:
        reasons.append("HTF structure undefined")

    last = st.last_event
    if last is not None and last.is_mss:
        if last.direction == "bull":
            score += 1
            reasons.append("confirmed by bullish MSS")
        else:
            score -= 1
            reasons.append("confirmed by bearish MSS")

    rng = dealing_range(df, swing_length=swing_length)
    pos = zone = None
    if rng is not None and not df.empty:
        last_close = float(df["close"].iloc[-1])
        pos = rng.position(last_close)
        zone = rng.zone(last_close)
        reasons.append(f"price in {zone} ({pos:.0%} of range)")

    direction = "bull" if score > 0 else "bear" if score < 0 else "neutral"
    return Bias(direction=direction, score=score, reasons=reasons,
                range_position=pos, range_zone=zone)
