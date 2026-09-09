"""Market intraday momentum (Baltussen, Da, Lammers, Martens, JFE 2021) on XAUUSD.

Claim: the return from the previous close to shortly before the close predicts the
return over the last 30-60 minutes of the session (gamma-hedging flow). Tested on
60+ futures incl. gold. COMEX gold settles 13:30 ET = 17:30 UTC (18:30 UTC in winter).

We test on M15 mid (2022+) and H1 (2017+):
  predictor: r_day = log(P[t_pre] / P[prev settle])
  target:    r_last = log(P[settle] / P[t_pre])
  strategy:  position = sign(r_day) for the last window, one round-trip spread per day.
Also the JFE variant: predictor = first-hour return only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from fsp.research.gold.data import load_mid_with_spread
from fsp.research.gold.evaluate import nw_tstat


def settle_utc(ts: pd.Timestamp) -> int:
    """17 during US DST (Mar-Nov), 18 otherwise: hour whose end is 13:30 ET +/- 30m."""
    et = ts.tz_convert("America/New_York")
    return 17 if et.dst() != pd.Timedelta(0) else 18


def build(df: pd.DataFrame, last_minutes: int) -> pd.DataFrame:
    d = df.copy()
    d["day"] = d.index.normalize()
    rows = []
    for day, g in d.groupby("day"):
        if len(g) < 20:
            continue
        s_h = settle_utc(g.index[0])
        settle = day + pd.Timedelta(hours=s_h, minutes=30)
        pre = settle - pd.Timedelta(minutes=last_minutes)
        first_end = day + pd.Timedelta(hours=1)
        try:
            p_settle = g.loc[:settle]["close"].iloc[-1]
            p_pre = g.loc[:pre]["close"].iloc[-1]
            p_open = g["open"].iloc[0]
            p_first = g.loc[:first_end]["close"].iloc[-1]
            spr = g.loc[:pre]["spread"].iloc[-1] / p_pre
        except IndexError:
            continue
        if g.loc[:pre].index[-1] < pre - pd.Timedelta(hours=2):
            continue  # holiday / half day
        rows.append({"day": day, "r_day": np.log(p_pre / p_open), "r_first": np.log(p_first / p_open),
                     "r_last": np.log(p_settle / p_pre), "cost": spr})
    return pd.DataFrame(rows).set_index("day")


def report(t: pd.DataFrame, label: str):
    for pred in ("r_day", "r_first"):
        x, y = t[pred], t["r_last"]
        rho = spearmanr(x, y).correlation
        beta = np.cov(x, y)[0, 1] / x.var()
        net = np.sign(x) * y - t["cost"]
        gross = np.sign(x) * y
        print(f"{label} pred={pred:8s} n={len(t)} rho={rho:+.3f} beta={beta:+.3f} "
              f"| gross {gross.mean()*1e4:+.2f} bps/day t={nw_tstat(gross.to_numpy(),0):+.2f} "
              f"| net {net.mean()*1e4:+.2f} bps/day t={nw_tstat(net.to_numpy(),0):+.2f} "
              f"Sharpe {net.mean()/net.std()*np.sqrt(252):+.2f} hit {(gross>0).mean()*100:.1f}%")
    # conditional on |r_day| large (paper: effect stronger on big-move days)
    x = t["r_day"]; big = x.abs() > x.abs().rolling(60).quantile(0.7).shift(1)
    tb = t[big.fillna(False)]
    net = np.sign(tb["r_day"]) * tb["r_last"] - tb["cost"]
    print(f"{label} |r_day| top-30% days: n={len(tb)} net {net.mean()*1e4:+.2f} bps/day t={nw_tstat(net.to_numpy(),0):+.2f}")
    yr = (np.sign(t["r_day"]) * t["r_last"] - t["cost"]).groupby(t.index.year)
    print("   per-year net bps/day:", {k: round(v * 1e4, 2) for k, v in yr.mean().items()})


def main():
    m15 = load_mid_with_spread("M15")
    for lm in (30, 60):
        t = build(m15, lm); report(t, f"M15 last{lm}m")
    h1 = load_mid_with_spread("H1"); h1 = h1[h1.index >= "2017-01-01"]
    t = build(h1, 60); report(t, "H1  last60m")


if __name__ == "__main__":
    main()
