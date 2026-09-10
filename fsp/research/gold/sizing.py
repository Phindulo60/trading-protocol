"""Dynamic lot sizing + account growth simulation for an R-multiple trade stream.

Given a sequence of trade outcomes in R (from any backtest) and their $/oz risk
distance, size each trade as a fixed fraction of CURRENT equity, snapped to the
broker's lot grid, bounded [min_lot, max_lot]. This is what 'dynamically up to
0.10 lots' means in practice: lots = floor(equity * risk_pct / (risk_$per_oz * 100) / 0.01) * 0.01.

Also: a drawdown throttle (halve risk after an X% peak-to-trough drawdown until
a new high) and a block-bootstrap Monte Carlo of the R stream so we report
distributions, not one path.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

OZ_PER_LOT = 100.0


@dataclass
class SizingConfig:
    start_equity: float = 1000.0
    risk_pct: float = 0.01          # fraction of equity risked per trade
    min_lot: float = 0.01
    max_lot: float = 0.10
    lot_step: float = 0.01
    margin_pct: float = 0.05        # broker margin on notional
    dd_throttle: float = 0.10       # halve risk when drawdown from peak exceeds this
    stop_equity: float = 300.0      # below this the account is considered dead
    commission_per_lot: float = 0.0


def size_lots(equity: float, risk_per_oz: float, cfg: SizingConfig, price: float) -> float:
    if risk_per_oz <= 0:
        return 0.0
    raw = equity * cfg.risk_pct / (risk_per_oz * OZ_PER_LOT)
    lots = np.floor(raw / cfg.lot_step) * cfg.lot_step
    lots = min(max(lots, 0.0), cfg.max_lot)
    # margin check: cannot open what we cannot margin
    if lots * OZ_PER_LOT * price * cfg.margin_pct > equity * 0.8:
        lots = np.floor(equity * 0.8 / (OZ_PER_LOT * price * cfg.margin_pct) / cfg.lot_step) * cfg.lot_step
    if lots < cfg.min_lot:
        # too small to trade at the target risk: either skip (0) or trade the minimum if it
        # risks no more than 3x the target (retail accounts have to accept some lumpiness)
        min_risk_frac = cfg.min_lot * OZ_PER_LOT * risk_per_oz / equity
        return cfg.min_lot if min_risk_frac <= 3 * cfg.risk_pct else 0.0
    return float(round(lots, 2))


def simulate_path(r: np.ndarray, risk_per_oz: np.ndarray, price: np.ndarray, cfg: SizingConfig) -> pd.DataFrame:
    eq = cfg.start_equity; peak = eq
    out = []
    for ri, rk, px in zip(r, risk_per_oz, price):
        throttled = (peak - eq) / peak > cfg.dd_throttle
        c = SizingConfig(**{**cfg.__dict__, "risk_pct": cfg.risk_pct * (0.5 if throttled else 1.0)})
        lots = size_lots(eq, rk, c, px)
        pnl = lots * OZ_PER_LOT * rk * ri - lots * cfg.commission_per_lot
        eq += pnl; peak = max(peak, eq)
        out.append((lots, pnl, eq, throttled))
        if eq < cfg.stop_equity:
            break
    return pd.DataFrame(out, columns=["lots", "pnl", "equity", "throttled"])


def monte_carlo(trades: pd.DataFrame, cfg: SizingConfig, n_paths: int = 5000, block: int = 8,
                horizon: int | None = None, seed: int = 0) -> dict:
    """Block-bootstrap the (R, risk$, price) triples preserving their joint distribution."""
    rng = np.random.default_rng(seed)
    r = trades["r_multiple"].to_numpy(); rk = (trades["fill"] - trades["sl"]).abs().to_numpy(); px = trades["fill"].to_numpy()
    n = len(r); H = horizon or n
    finals, maxdd, dead, weekly = [], [], 0, []
    for _ in range(n_paths):
        idx = np.concatenate([np.arange(s, s + block) for s in rng.integers(0, n - block, size=H // block + 1)])[:H] % n
        p = simulate_path(r[idx], rk[idx], px[idx], cfg)
        e = p["equity"].to_numpy()
        finals.append(e[-1]); maxdd.append((e / np.maximum.accumulate(np.r_[cfg.start_equity, e])[1:] - 1).min())
        dead += int(e[-1] < cfg.stop_equity)
    finals = np.array(finals); maxdd = np.array(maxdd)
    return {"n_trades": H, "median_final": float(np.median(finals)), "p10_final": float(np.percentile(finals, 10)),
            "p90_final": float(np.percentile(finals, 90)), "P(final>start)": float((finals > cfg.start_equity).mean()),
            "P(double)": float((finals > 2 * cfg.start_equity).mean()), "P(dead)": dead / n_paths,
            "median_maxDD": float(np.median(maxdd)), "p90_maxDD": float(np.percentile(maxdd, 10))}
