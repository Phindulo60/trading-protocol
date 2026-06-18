"""Draw-on-liquidity (DOL) targeting + sweep significance.

ICT 2022 Mentorship Ch.5/27: price gravitates to *standing* liquidity — the
previous day/week/month high & low — daily. "Bullish order flow seeks buyside
liquidity above the previous day's high." Those levels are magnets; targeting
them (rather than a minor structural swing) gives a more reliable draw and, in a
low-win-rate / high-R model, better realised reward.

Sweep significance is the Turtle-Soup tell: a reversal that *runs a major
standing level* (PDH/PDL/PWH/...) is higher-conviction than one that merely
pierces a fresh minor swing.

This builds on ``fsp.context.levels.htf_levels`` (PD/PW/PM H&L) and the
structural pools from ``fsp.ict.liquidity``.
"""
from __future__ import annotations

from fsp.context.sessions import DEFAULT_TZ
from fsp.context.levels import htf_levels, mark_swept
from fsp.data.types import Level
from fsp.ict.liquidity import LiquidityPool, nearest_unswept

import pandas as pd

# significance rank for standing levels (monthly > weekly > daily)
LEVEL_RANK = {"PM": 3, "PW": 2, "PD": 1}


def significant_levels(df: pd.DataFrame, tz: str = DEFAULT_TZ) -> dict[str, Level]:
    """PDH/PDL/PWH/PWL/PMH/PML for `df`, marked swept. Opens are dropped (they
    aren't liquidity targets)."""
    lv = htf_levels(df, tz)
    lv = {k: v for k, v in lv.items() if (k.endswith("H") or k.endswith("L"))
          and k[:2] in LEVEL_RANK}
    return mark_swept(lv, df)


def draw_targets(
    pools: list[LiquidityPool],
    levels: dict[str, Level] | None,
    side: str,           # "buy" -> draws above (long TP), "sell" -> below (short TP)
    entry: float,
) -> list[tuple[float, str]]:
    """Unswept *significant* draws beyond entry, nearest first.

    Significant = a standing HTF level (PDH/PWH/...) or a relative-equal pool.
    Minor single swings are intentionally excluded here (they're the fallback).
    """
    out: list[tuple[float, str]] = []
    for lbl, lv in (levels or {}).items():
        if lv.swept:
            continue
        if side == "buy" and lv.kind == "high" and lv.price > entry:
            out.append((lv.price, lbl))
        elif side == "sell" and lv.kind == "low" and lv.price < entry:
            out.append((lv.price, lbl))
    for p in pools:
        if p.swept or p.kind != "equal":
            continue
        if side == "buy" and p.side == "buy" and p.price > entry:
            out.append((p.price, "EQH"))
        elif side == "sell" and p.side == "sell" and p.price < entry:
            out.append((p.price, "EQL"))
    out.sort(key=lambda c: abs(c[0] - entry))
    return out


def best_target(
    pools: list[LiquidityPool],
    levels: dict[str, Level] | None,
    side: str,
    entry: float,
) -> tuple[float | None, str | None]:
    """Pick the draw-on-liquidity target: nearest significant level/equal-pool
    beyond entry; fall back to nearest structural swing pool. Returns
    (price, label)."""
    sig = draw_targets(pools, levels, side, entry)
    if sig:
        return sig[0]
    swing = nearest_unswept(pools, side, entry)
    if swing is not None:
        return swing.price, "swing"
    return None, None


def sweep_significance(
    sweep_level: float,
    sweep_side: str,         # "buy" = ran a high pool, "sell" = ran a low pool
    levels: dict[str, Level] | None,
    tol: float,
) -> str | None:
    """Label of the major standing level the sweep ran (within `tol`), else None.
    A non-None result is the canonical Turtle-Soup high-conviction reversal."""
    want_kind = "high" if sweep_side == "buy" else "low"
    best = None
    best_rank = -1
    for lbl, lv in (levels or {}).items():
        if lv.kind != want_kind:
            continue
        if abs(lv.price - sweep_level) <= tol:
            rank = LEVEL_RANK.get(lbl[:2], 0)
            if rank > best_rank:
                best, best_rank = lbl, rank
    return best
