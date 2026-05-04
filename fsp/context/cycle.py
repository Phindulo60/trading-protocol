"""Market cycle classifier: EXPANSION / CONSOLIDATION / NEUTRAL.

Heuristic combining:
  - ATR(5) vs ATR(50): fast > slow*1.15 = expansion, < slow*0.85 = consolidation
  - Range compression: rolling 20-bar range < 70% of ADR(5) suggests consolidation
  - Current daily range vs 5-day ADR (context)
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from fsp.data.types import Cycle
from fsp.structure.displacement import atr


@dataclass
class CycleState:
    cycle: Cycle
    atr_fast: float
    atr_slow: float
    atr_ratio: float
    adr5: float
    today_range: float
    adr_pct: float  # 0–100+


def classify_cycle(df: pd.DataFrame, daily_df: pd.DataFrame | None = None) -> CycleState:
    if len(df) < 50:
        return CycleState(Cycle.NEUTRAL, 0, 0, 1.0, 0, 0, 0)

    af = atr(df, 5).iloc[-1]
    as_ = atr(df, 50).iloc[-1]
    ratio = float(af / as_) if as_ else 1.0

    # Daily ADR / today's range
    if daily_df is not None and len(daily_df) >= 6:
        adr5 = float((daily_df["high"] - daily_df["low"]).tail(6).head(5).mean())
        today = daily_df.iloc[-1]
        today_range = float(today["high"] - today["low"])
    else:
        adr5 = float(((df["high"] - df["low"]).rolling(24).sum()).tail(5).mean())
        today_range = float((df["high"] - df["low"]).tail(24).sum())
    adr_pct = (today_range / adr5 * 100) if adr5 else 0.0

    if ratio > 1.15:
        cyc = Cycle.EXPANSION
    elif ratio < 0.85:
        cyc = Cycle.CONSOLIDATION
    else:
        cyc = Cycle.NEUTRAL

    return CycleState(cyc, float(af), float(as_), ratio, adr5, today_range, adr_pct)
