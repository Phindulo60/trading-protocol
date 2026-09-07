"""Gold's return accrues in the Asia reopen window (21:00-00:00 UTC).

Test: long from the open of the first bar >= `h0` UTC to the close of the
23:00 bar, every trading day. One round-trip spread per day. Per-year table
so a single regime cannot carry the result.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fsp.research.gold.data import load_mid_with_spread
from fsp.research.gold.evaluate import nw_tstat


def window_returns(df: pd.DataFrame, h0: int, h1: int = 23) -> pd.DataFrame:
    """Per-day log return open(first bar with hour>=h0) -> close(bar hour==h1), net of spread."""
    d = df.copy()
    d["day"] = d.index.normalize()
    d["hour"] = d.index.hour
    win = d[(d["hour"] >= h0) & (d["hour"] <= h1)]
    g = win.groupby("day")
    first = g.first(); last = g.last()
    out = pd.DataFrame({
        "gross": np.log(last["close"] / first["open"]),
        "cost": (first["spread"] / first["open"]),
        "bars": g.size(),
    })
    out["net"] = out["gross"] - out["cost"]
    return out[out["bars"] >= (h1 - h0)]  # drop truncated days (holidays)


def rest_of_day(df: pd.DataFrame, h0: int) -> pd.Series:
    d = df.copy(); d["day"] = d.index.normalize(); d["hour"] = d.index.hour
    win = d[d["hour"] < h0]; g = win.groupby("day")
    return np.log(g.last()["close"] / g.first()["open"])


def main():
    df = load_mid_with_spread("H1")
    df = df[df.index >= "2017-01-01"]
    for h0 in (21, 22, 23):
        w = window_returns(df, h0)
        print(f"\n=== long {h0}:00 -> 00:00 UTC, n={len(w)} days ===")
        rows = []
        for yr, g in w.groupby(w.index.year):
            rows.append({"year": yr, "days": len(g), "gross_bps": g.gross.mean() * 1e4,
                         "net_bps": g.net.mean() * 1e4, "net_total_pct": g.net.sum() * 100,
                         "t_net": nw_tstat(g.net.to_numpy(), 0), "win%": (g.net > 0).mean() * 100})
        print(pd.DataFrame(rows).set_index("year").round(2).to_string())
        net = w.net.to_numpy()
        ann = net.mean() / net.std(ddof=1) * np.sqrt(252)
        print(f"ALL: gross {w.gross.mean()*1e4:.2f} bps/day, net {net.mean()*1e4:.2f} bps/day, "
              f"t={nw_tstat(net, 0):.2f}, Sharpe={ann:.2f}, total net {net.sum()*100:.1f}%")
        rod = rest_of_day(df, h0).reindex(w.index)
        print(f"     rest-of-day (00:00->{h0}:00) gross {rod.mean()*1e4:.2f} bps/day, t={nw_tstat(rod.dropna().to_numpy(), 0):.2f}")
    # buy-and-hold over same period for reference
    bh = np.log(df["close"].iloc[-1] / df["close"].iloc[0]) * 100
    print(f"\nbuy-and-hold same period: {bh:.1f}% ; window captures share above")


if __name__ == "__main__":
    main()
