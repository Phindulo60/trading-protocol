"""Replication of arXiv:2511.08571 'Forecast-to-Fill' gold trend/momentum regime strategy.

Paper (daily spot gold, 2015-2025 OOS, rolling 10y train / 6m test):
  y = log P; EMA smoothing y~_t = lam*y~_{t-1} + (1-lam)*y_t ; slope = diff(y~)
  z = (slope - mu_train)/sd_train, clip [-3,3]; p_trend = (z+3)/6
  m = 1{P_t / P_{t-50} > 1};  p_bull = 0.6*p_trend + 0.4*m
  long if p_bull >= 0.52 and slope > 0
  exits: hard stop entry-2*ATR14, trailing peak-1.5*ATR14, timeout 30d, de-risk if p_bear>0.5
  sizing: EWMA vol target 15% ann, cap 2x, x regime share, x fractional Kelly 0.4 (baseline 25%)
  cost: 0.7 bps linear.
Reported: Sharpe 2.88, realised vol 0.91%, CAGR 2.65%, maxDD 0.52%, hit-rate 65.8%.

We run the rule as specified on Dukascopy daily mid, cost = observed spread (2.3 bps, 3x the paper),
and ALSO with the paper's 0.7 bps, over the paper's OOS window and ours. Lambda is not stated
numerically in the text we could access; we sweep it and report every value (no cherry-pick).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fsp.research.gold.data import load_mid_with_spread
from fsp.research.gold.evaluate import nw_tstat


def atr(df, n=14):
    h, l, c = df["high"], df["low"], df["close"].shift()
    tr = pd.concat([h - l, (h - c).abs(), (l - c).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def run(df: pd.DataFrame, lam: float, cost_bps: float | None, train_years: int = 10,
        test_months: int = 6, K: int = 50, omega: float = 0.6, thr: float = 0.52,
        vol_target: float = 0.15, wmax: float = 2.0, kelly_frac: float = 0.4,
        use_stops: bool = True) -> pd.DataFrame:
    c = df["close"]; y = np.log(c)
    ysm = y.ewm(alpha=1 - lam, adjust=False).mean()
    slope = ysm.diff()
    mom = (c / c.shift(K) > 1).astype(float)
    ret = c.pct_change()
    a14 = atr(df)
    # EWMA vol forecast (RiskMetrics 0.94), uses returns up to t for position at t+1
    var = (ret ** 2).ewm(alpha=1 - 0.94, adjust=False).mean()
    sig_next = np.sqrt(var)
    spread = (df["spread"] / c) if cost_bps is None else pd.Series(cost_bps / 1e4, index=df.index)

    out = []
    start = df.index[0] + pd.DateOffset(years=train_years)
    t0 = start
    pos = 0.0; entry = np.nan; peak = np.nan; age = 0
    while t0 < df.index[-1]:
        t1 = t0 + pd.DateOffset(months=test_months)
        tr = slope[(slope.index < t0) & (slope.index >= t0 - pd.DateOffset(years=train_years))].dropna()
        mu, sd = tr.mean(), tr.std()
        # Kelly on training window: unit-notional strategy return when rule is long
        p_tr = ((slope[tr.index] - mu) / sd).clip(-3, 3).add(3).div(6)
        pb_tr = omega * p_tr + (1 - omega) * mom[tr.index]
        sig_tr = ((pb_tr >= thr) & (slope[tr.index] > 0)).astype(float).shift(1)
        R = (sig_tr * ret[tr.index]).dropna()
        R = R[sig_tr.reindex(R.index) > 0]
        k_cost = float(spread[tr.index].mean())
        f_star = max((R.mean() - k_cost) / R.var(), 0.0) if len(R) > 20 and R.var() > 0 else 0.0
        f_kelly = kelly_frac * f_star
        test = df.index[(df.index >= t0) & (df.index < t1)]
        for t in test:
            i = df.index.get_loc(t)
            if i == 0 or not np.isfinite(slope.iloc[i - 1]):
                continue
            # decision uses info through t-1, applied to return t
            z = np.clip((slope.iloc[i - 1] - mu) / sd, -3, 3)
            p_bull = omega * (z + 3) / 6 + (1 - omega) * mom.iloc[i - 1]
            w_vol = min(vol_target / np.sqrt(252) / max(sig_next.iloc[i - 1], 1e-6), wmax)
            share = np.clip((p_bull - 0.5) / 0.5, 0, 1)
            size = w_vol * share * (f_kelly if f_kelly > 0.01 else 0.25)
            size = min(size, wmax)
            new_pos = pos
            if pos == 0 and p_bull >= thr and slope.iloc[i - 1] > 0:
                new_pos = size; entry = c.iloc[i - 1]; peak = entry; age = 0
            elif pos > 0:
                age += 1
                exit_ = False
                if use_stops:
                    if c.iloc[i - 1] < entry - 2 * a14.iloc[i - 1]: exit_ = True
                    if c.iloc[i - 1] < peak - 1.5 * a14.iloc[i - 1]: exit_ = True
                    if age >= 30: exit_ = True
                if 1 - p_bull > 0.5: new_pos = pos * 0.5
                if exit_: new_pos = 0.0
                else: new_pos = min(new_pos if new_pos != pos else size, wmax) if not exit_ else 0.0
            turnover = abs(new_pos - pos)
            r_strat = new_pos * ret.iloc[i] - turnover * spread.iloc[i]
            pos = new_pos
            if pos > 0: peak = max(peak, c.iloc[i])
            out.append((t, r_strat, pos, ret.iloc[i]))
        t0 = t1
    res = pd.DataFrame(out, columns=["ts", "r", "w", "gold"]).set_index("ts")
    return res


def summarize(res: pd.DataFrame, label: str):
    r = res["r"]
    ann = r.mean() * 252; vol = r.std() * np.sqrt(252)
    eq = (1 + r).cumprod(); dd = (eq / eq.cummax() - 1).min()
    active = res["w"] > 0
    hit = (r[active] > 0).mean() if active.any() else np.nan
    beta = np.cov(r, res["gold"])[0, 1] / res["gold"].var()
    print(f"{label:34s} Sharpe {ann/vol if vol>0 else np.nan:5.2f}  CAGR {(eq.iloc[-1]**(252/len(r))-1)*100:6.2f}%  "
          f"vol {vol*100:5.2f}%  maxDD {dd*100:6.2f}%  hit {hit*100:4.1f}%  time-in {active.mean()*100:4.1f}%  "
          f"mean|w| {res.w.abs().mean():.3f}  beta {beta:.2f}  t={nw_tstat(r.to_numpy(),0):.2f}")


def main():
    df = load_mid_with_spread("D")
    df.index = df.index.normalize(); df = df[~df.index.duplicated(keep="last")]
    df = df[df.index >= "2010-01-01"]
    print(f"data {df.index[0].date()}..{df.index[-1].date()} n={len(df)}  median spread {float((df.spread/df.close).median()*1e4):.2f} bps")
    gold = df["close"].pct_change()
    for label, tr_y in (("paper-like: 10y train (OOS 2020+)", 10), ("5y train (OOS 2015+)", 5)):
        print(f"\n== {label} ==")
        for lam in (0.90, 0.94, 0.97):
            for cost in (0.7, None):
                res = run(df, lam, cost, train_years=tr_y)
                summarize(res, f"lam={lam} cost={'0.7bps' if cost else 'observed'}")
        res = run(df, 0.94, None, train_years=tr_y, use_stops=False)
        summarize(res, "lam=0.94 observed, NO stops")
        g = gold[res.index]
        eq = (1 + g).cumprod()
        print(f"{'buy&hold gold same window':34s} Sharpe {g.mean()/g.std()*np.sqrt(252):5.2f}  CAGR {(eq.iloc[-1]**(252/len(g))-1)*100:6.2f}%  vol {g.std()*np.sqrt(252)*100:5.2f}%  maxDD {(eq/eq.cummax()-1).min()*100:6.2f}%")
    # per-year for the paper-like config
    res = run(df, 0.94, 0.7, train_years=10)
    print("\nper-year (lam=0.94, 0.7bps, 10y train):")
    yr = res["r"].groupby(res.index.year)
    print(pd.DataFrame({"ret%": yr.sum()*100, "sharpe": yr.apply(lambda x: x.mean()/x.std()*np.sqrt(252) if x.std()>0 else np.nan),
                        "time_in%": res["w"].gt(0).groupby(res.index.year).mean()*100}).round(2).to_string())


if __name__ == "__main__":
    main()
