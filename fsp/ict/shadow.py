"""ICT shadow signal path.

Runs the confluence engine in real time alongside the live strategies and emits
a ``Signal(strategy="ICT_SHADOW")`` for every A/A+ setup. These are *logged to
the journal only* (sent=False) — never alerted, never traded — so the resolver
can stamp real forward outcomes and we can compare the ICT engine's live edge
against the deployed strategies before risking anything.
"""
from __future__ import annotations

import pandas as pd

from fsp.signals.base import Signal
from fsp.ict.engine import decide, TradeDecision
from fsp.ict.backtest import GRADE_RANK


def decision_to_signal(d: TradeDecision, pair: str) -> Signal | None:
    """Map a tradable TradeDecision onto the shared Signal schema."""
    if not d.is_tradable:
        return None
    pip = 0.01 if "JPY" in pair else 0.0001
    inv_pips = abs(d.entry - d.stop) / pip
    note = f"ICT {d.grade} {d.score}/12: " + ", ".join(d.confluences[:4])
    return Signal(
        strategy="ICT_SHADOW",
        pair=pair,
        direction=d.direction,
        entry=float(d.entry),
        sl=float(d.stop),
        tp1=float(d.target),
        tp2=None,
        inv_pips=round(inv_pips, 1),
        rr_tp1=round(float(d.rr), 2) if d.rr is not None else 0.0,
        rr_tp2=None,
        risk_r=1.0,
        note=note,
        ts=pd.Timestamp(d.ts).isoformat(),
        context={
            "grade": d.grade, "score": d.score, "htf_bias": d.htf_bias,
            "confluences": d.confluences, "missing": d.missing,
        },
    )


def scan_ict_shadow(
    pair: str,
    m15_df: pd.DataFrame,
    h1_df: pd.DataFrame | None = None,
    *,
    min_grade: str = "A",
    window: int = 250,
    htf_window: int = 180,
    swing_length: int = 5,
    lookback: int = 30,
    atr_mult: float = 1.5,
    atr_len: int = 20,
    drop_forming_htf: bool = True,
) -> Signal | None:
    """Evaluate the latest M15 bar for an ICT setup. Returns a shadow Signal or None."""
    if m15_df is None or len(m15_df) < max(window // 2, 2 * swing_length + 1):
        return None
    ltf = m15_df.tail(window)
    htf = None
    if h1_df is not None and len(h1_df) > 1:
        h = h1_df.iloc[:-1] if drop_forming_htf else h1_df   # drop the forming HTF bar
        htf = h.tail(htf_window)
    try:
        d = decide(ltf, htf, pair=pair, swing_length=swing_length, lookback=lookback,
                   atr_mult=atr_mult, atr_len=atr_len)
    except Exception:
        return None
    if not d.is_tradable:
        return None
    if GRADE_RANK.get(d.grade, 0) < GRADE_RANK.get(min_grade, 2):
        return None
    return decision_to_signal(d, pair)


def scan_batch_ict_shadow(
    batch: dict[str, tuple[pd.DataFrame, pd.DataFrame | None]],
    *,
    min_grade: str = "A",
    **kw,
) -> list[Signal]:
    """Run the shadow scan over {pair: (m15_df, h1_df)} and return fired signals."""
    out: list[Signal] = []
    for pair, (m15_df, h1_df) in batch.items():
        sig = scan_ict_shadow(pair, m15_df, h1_df, min_grade=min_grade, **kw)
        if sig is not None:
            out.append(sig)
    return out
