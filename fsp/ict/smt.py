"""SMT (Smart Money Technique) divergence — cross-pair non-confirmation.

ICT 2022 Mentorship, Ch.17 (Intermarket Relationships): correlated instruments
should make matching highs/lows. When one runs a liquidity level but its
correlate does **not** confirm — a *divergence* — the run is a stop-raid by smart
money, not a genuine move, and price tends to reverse.

We anchor the comparison to a *sweep*: the swept pool's origin swing (``ref_ts``)
and the sweep bar (``sweep_ts``). Between those two points the primary made a
new extreme (the raid). If the partner failed to make a matching new extreme,
SMT divergence is present and the reversal thesis is corroborated.

Correlation is encoded with a sign: +1 positively correlated (move together),
-1 inversely correlated (mirror). For an inverse partner we read the opposite
extreme, so the divergence test stays uniform.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

# pair -> (partner, sign). Same-family majors are the cleanest correlations:
#   USD-quote (EUR/GBP/AUD vs USD) move together; JPY crosses move together;
#   USD-base (USDCAD/USDJPY) move together (driven by USD strength).
PARTNERS: dict[str, tuple[str, int]] = {
    "EURUSD": ("GBPUSD", 1),
    "GBPUSD": ("EURUSD", 1),
    "AUDUSD": ("EURUSD", 1),
    "USDCAD": ("USDJPY", 1),
    "USDJPY": ("USDCAD", 1),
    "EURJPY": ("GBPJPY", 1),
    "GBPJPY": ("EURJPY", 1),
}


def partner_for(pair: str) -> tuple[str, int] | None:
    """Return (partner_pair, sign) for SMT, or None if we have no clean pairing."""
    return PARTNERS.get(pair)


@dataclass
class SMTResult:
    diverged: bool
    direction: str               # "long" (primary swept lows) / "short" (swept highs)
    sign: int                    # +1 positive corr, -1 inverse
    partner: str | None
    ref: float | None            # partner extreme at the pool-origin reference
    now: float | None            # partner extreme at the sweep
    note: str = ""


def _extreme(df: pd.DataFrame | None, ts, window: int, kind: str) -> float | None:
    """Partner's local low/high within +/-`window` bars of `ts`."""
    if df is None or len(df) == 0:
        return None
    pos = df.index.searchsorted(pd.Timestamp(ts))
    lo = max(0, pos - window)
    hi = min(len(df), pos + window + 1)
    seg = df.iloc[lo:hi]
    if seg.empty:
        return None
    return float(seg["low"].min()) if kind == "low" else float(seg["high"].max())


def smt_divergence(
    primary_df: pd.DataFrame,
    partner_df: pd.DataFrame | None,
    *,
    ref_ts,
    sweep_ts,
    direction: str,
    sign: int = 1,
    partner: str | None = None,
    window: int = 3,
) -> SMTResult:
    """Test whether `partner_df` failed to confirm the primary's raid.

    direction="long": primary swept *lows* (a lower low). A positively-correlated
    partner confirms by also making a lower low; SMT divergence = it made a
    *higher* low instead. (Mirror for shorts; inverse sign reads the other
    extreme so the comparison is symmetric.)
    """
    if direction == "long":
        kind = "low" if sign > 0 else "high"
    else:
        kind = "high" if sign > 0 else "low"

    ref = _extreme(partner_df, ref_ts, window, kind)
    now = _extreme(partner_df, sweep_ts, window, kind)
    if ref is None or now is None:
        return SMTResult(False, direction, sign, partner, ref, now, "insufficient partner data")

    if direction == "long":
        diverged = (now > ref) if sign > 0 else (now < ref)
    else:
        diverged = (now < ref) if sign > 0 else (now > ref)
    return SMTResult(diverged, direction, sign, partner, ref, now,
                     "divergence" if diverged else "confirmation")
