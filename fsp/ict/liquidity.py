"""ICT liquidity model: pools + reclaim-sweep detection.

Liquidity rests where stop orders cluster — above old/equal highs (buy-side)
and below old/equal lows (sell-side). Price is drawn to these pools, runs them
(a *sweep* / stop-run), and frequently reverses. The high-probability tell is a
**reclaim sweep**: a bar that wicks beyond the pool but closes back inside it.

  - sell-side sweep  (low pierces a low pool, close back above)   -> bullish
  - buy-side  sweep  (high pierces a high pool, close back below) -> bearish

This complements ``fsp.context.levels`` (time-based PDH/PDL/PWH/...). Here the
pools are *structural* — built from confirmed swing pivots and clustered into
relative-equal pools.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

import numpy as np
import pandas as pd

from fsp.data.types import Swing
from fsp.structure.swings import find_swings
from fsp.structure.displacement import atr


@dataclass
class LiquidityPool:
    price: float
    side: Literal["buy", "sell"]        # buy-side (above price) / sell-side (below)
    kind: Literal["equal", "swing"]     # relative-equal cluster vs single swing
    members: list[datetime]             # swing timestamps composing the pool
    created_ts: datetime                # last member ts — pool is live after this
    swept: bool = False
    swept_ts: datetime | None = None

    @property
    def strength(self) -> int:
        """Number of swings stacked at this level (more = more liquidity)."""
        return len(self.members)


@dataclass
class LiquiditySweep:
    ts: datetime                        # the sweep bar
    side: Literal["buy", "sell"]        # which liquidity was taken
    direction: Literal["bull", "bear"]  # reversal implication of the reclaim
    level: float                        # pool price that was run
    extreme: float                      # wick extreme (the stop-run high/low)
    close: float                        # close back inside the level
    pool: LiquidityPool
    kind: Literal["equal", "swing"]


def _tol(df: pd.DataFrame, eq_tol_atr: float, atr_len: int) -> float:
    """Absolute price tolerance for 'relative equal' clustering.

    Prefers the median ATR; falls back to the mean candle range so short frames
    (where ATR is still NaN) still get a sane tolerance.
    """
    a = atr(df, atr_len).to_numpy()
    amed = np.nanmedian(a) if np.isfinite(a).any() else np.nan
    if not np.isfinite(amed) or amed == 0:
        amed = float((df["high"] - df["low"]).mean())
    return float(eq_tol_atr * amed)


def _cluster(swings: list[Swing], tol: float, want_max: bool) -> list[LiquidityPool]:
    """Cluster consecutive same-kind swings within `tol` into pools."""
    pools: list[LiquidityPool] = []
    cur: list[Swing] = []

    def flush():
        if not cur:
            return
        prices = [s.price for s in cur]
        price = max(prices) if want_max else min(prices)
        pools.append(LiquidityPool(
            price=price,
            side="buy" if want_max else "sell",
            kind="equal" if len(cur) >= 2 else "swing",
            members=[s.ts for s in cur],
            created_ts=cur[-1].ts,
        ))

    for s in swings:
        if not cur:
            cur = [s]
            continue
        ref = sum(x.price for x in cur) / len(cur)
        if abs(s.price - ref) <= tol:
            cur.append(s)
        else:
            flush()
            cur = [s]
    flush()
    return pools


def find_liquidity_pools(
    df: pd.DataFrame,
    swing_length: int = 5,
    eq_tol_atr: float = 0.1,
    atr_len: int = 20,
) -> list[LiquidityPool]:
    """Build buy-side and sell-side liquidity pools from confirmed swings."""
    swings = find_swings(df, length=swing_length)
    if not swings:
        return []
    tol = _tol(df, eq_tol_atr, atr_len)
    highs = [s for s in swings if s.kind == "high"]
    lows = [s for s in swings if s.kind == "low"]
    pools = _cluster(highs, tol, want_max=True) + _cluster(lows, tol, want_max=False)
    pools.sort(key=lambda p: p.created_ts)
    return pools


def find_sweeps(
    df: pd.DataFrame,
    pools: list[LiquidityPool] | None = None,
    swing_length: int = 5,
    eq_tol_atr: float = 0.1,
    buffer_atr: float = 0.0,
    atr_len: int = 20,
) -> list[LiquiditySweep]:
    """Detect the first reclaim-sweep of each pool.

    A pool is swept when, on a bar after it formed, price wicks beyond the level
    (+/- an optional ATR buffer) yet *closes back inside* — the classic stop-run
    rejection. Mutates each swept pool's ``swept``/``swept_ts``.
    """
    if pools is None:
        pools = find_liquidity_pools(df, swing_length, eq_tol_atr, atr_len)
    if not pools:
        return []

    a = atr(df, atr_len).to_numpy()
    amed = np.nanmedian(a) if np.isfinite(a).any() else np.nan
    if not np.isfinite(amed) or amed == 0:
        amed = float((df["high"] - df["low"]).mean())
    buf = float(buffer_atr * amed)

    sweeps: list[LiquiditySweep] = []
    for p in pools:
        fut = df[df.index > pd.Timestamp(p.created_ts)]
        if fut.empty:
            continue
        if p.side == "buy":
            reclaim = (fut["high"] > p.price + buf) & (fut["close"] < p.price)
            brk = fut["close"] > p.price          # closed above the high = consumed
        else:
            reclaim = (fut["low"] < p.price - buf) & (fut["close"] > p.price)
            brk = fut["close"] < p.price          # closed below the low = consumed
        r_hits, b_hits = fut[reclaim], fut[brk]
        if r_hits.empty:
            continue
        # the pool is taken only if the reclaim precedes any clean break-through
        if not b_hits.empty and b_hits.index[0] < r_hits.index[0]:
            continue
        ts = r_hits.index[0]
        row = r_hits.iloc[0]
        p.swept = True
        p.swept_ts = ts.to_pydatetime()
        sweeps.append(LiquiditySweep(
            ts=ts.to_pydatetime(),
            side=p.side,
            direction="bear" if p.side == "buy" else "bull",
            level=p.price,
            extreme=float(row["high"] if p.side == "buy" else row["low"]),
            close=float(row["close"]),
            pool=p,
            kind=p.kind,
        ))
    sweeps.sort(key=lambda s: s.ts)
    return sweeps


def nearest_unswept(
    pools: list[LiquidityPool],
    side: Literal["buy", "sell"],
    ref_price: float,
) -> LiquidityPool | None:
    """Closest unswept pool of `side` beyond `ref_price` — a draw-on-liquidity / TP target.

    buy-side -> nearest pool above ref_price; sell-side -> nearest below.
    """
    if side == "buy":
        cands = [p for p in pools if p.side == "buy" and not p.swept and p.price > ref_price]
        cands.sort(key=lambda p: p.price)
    else:
        cands = [p for p in pools if p.side == "sell" and not p.swept and p.price < ref_price]
        cands.sort(key=lambda p: -p.price)
    return cands[0] if cands else None
