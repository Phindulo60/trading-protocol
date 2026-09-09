"""What does leverage on the HAR vol-targeted gold long do to a $500 account?

Block-bootstrap (2-week blocks) of REAL daily strategy returns (OOS 2021-11..2026-09,
vol-target cap 2x, cost-adjusted) at account leverage L. TradeNation constraints:
min 0.01 lot = 1 oz ~ $4,400 notional, margin 5% (~$220). With $500 the max is 0.02 lot
(L ~ 17.6). Ruin = equity < margin for 0.01 lot (can no longer hold a position).
Reports per L: P(week >= +$50), median week, P(ruin within 12 weeks), P(ruin within 52 weeks).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from fsp.research.gold.data import load_mid_with_spread
from fsp.research.gold.vol_target import har_forecast


def strategy_returns() -> pd.Series:
    df = load_mid_with_spread("H1"); df = df[df.index >= "2017-01-01"]
    r = np.log(df["close"]).diff(); day = r.index.normalize()
    rv = (r ** 2).groupby(day).sum().pow(0.5); rv = rv[rv > 0]
    dclose = df["close"].groupby(day).last(); dret = np.log(dclose).diff()
    dspread = (df["spread"] / df["close"]).groupby(day).median()
    fc = har_forecast(rv); idx = fc.index.intersection(dret.index)
    target = float(rv[:fc.index[0]].mean())
    w = (target / fc[idx]).clip(upper=2.0).shift(1).fillna(1.0)
    turn = w.diff().abs().fillna(w.iloc[0])
    return (w * dret[idx] - turn * dspread[idx]).dropna()


def simulate(strat: pd.Series, L: float, capital: float = 500.0, weeks: int = 52,
             n_paths: int = 20000, block: int = 10, seed: int = 0,
             gold_px: float = 4411.0, margin_pct: float = 0.05, swap_bps_day: float = 0.0):
    rng = np.random.default_rng(seed)
    x = strat.to_numpy(); n = len(x); days = weeks * 5
    min_margin = gold_px * 1.0 * margin_pct  # 0.01 lot = 1 oz = $4,411 notional -> ~$220 margin
    starts = rng.integers(0, n - block, size=(n_paths, days // block + 1))
    paths = np.concatenate([x[s:s + block] for s in starts[0]])[:days]  # shape check
    sims = np.empty((n_paths, days))
    for i in range(n_paths):
        sims[i] = np.concatenate([x[s:s + block] for s in starts[i]])[:days]
    lev = L * sims - swap_bps_day / 1e4 * L  # daily account return
    eq = capital * np.cumprod(1 + lev, axis=1)
    # ruin: equity below margin for the smallest position at any point (position closed out / cannot re-enter)
    ruined = (eq < min_margin).any(axis=1)
    ruin12 = (eq[:, :60] < min_margin).any(axis=1)
    wk = eq[:, 4::5]; wk_prev = np.concatenate([np.full((n_paths, 1), capital), wk[:, :-1]], axis=1)
    wk_pnl = (wk - wk_prev)[:, :12]  # first 12 weeks, before survivorship bias kicks in
    alive12 = ~ruin12
    return {
        "L": L, "notional$": round(capital * L), "oz": round(capital * L / gold_px, 2),
        "P(week>=+50)%": (wk_pnl >= 50).mean() * 100,
        "median week $": np.median(wk_pnl), "p10 week $": np.percentile(wk_pnl, 10),
        "P(ruin 12wk)%": ruin12.mean() * 100, "P(ruin 52wk)%": ruined.mean() * 100,
        "median eq 52wk $": np.median(eq[:, -1]), "P(eq>1000 @52wk)%": (eq[:, -1] > 1000).mean() * 100,
    }


def main():
    s = strategy_returns()
    print(f"strategy daily returns: n={len(s)} mean {s.mean()*1e4:.1f} bps sd {s.std()*1e4:.0f} bps "
          f"Sharpe {s.mean()/s.std()*np.sqrt(252):.2f} worst day {s.min()*100:.1f}% worst 10d {s.rolling(10).sum().min()*100:.1f}%")
    rows = [simulate(s, L) for L in (1, 2, 5, 8.8, 17.6)]
    pd.set_option("display.width", 220)
    print(pd.DataFrame(rows).set_index("L").round(1).to_string())
    print("\nnote: 0.01 lot = 1 oz. L=8.8 is the MINIMUM tradeable position on $500; L=17.6 (0.02 lot) is the max margin allows.")
    print("      L<8.8 rows are hypothetical (would need ~$4,400+ of capital to trade at that leverage).")


if __name__ == "__main__":
    main()
