"""Does the HAR-RV vol forecast improve a long-gold position via sizing?

Constant long vs vol-targeted long (exposure = target_vol / forecast_vol,
capped at 2x), daily rebalance, cost = spread * |change in exposure|.
Same OOS period as diagnostics.har_rv. Sharpe, max drawdown, per-year.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from fsp.research.gold.data import load_mid_with_spread
from fsp.research.gold.evaluate import nw_tstat


def har_forecast(rv: pd.Series, start_frac: float = 0.5) -> pd.Series:
    lrv = np.log(rv)
    X = pd.DataFrame({"d": lrv.shift(1), "w": lrv.rolling(5).mean().shift(1),
                      "m": lrv.rolling(22).mean().shift(1)})
    m = X.notna().all(axis=1) & lrv.notna()
    X, y = X[m], lrv[m]
    n = len(y); cut = int(n * start_frac)
    preds = []
    for i in range(cut, n, 22):
        lr = LinearRegression().fit(X.iloc[:i], y.iloc[:i])
        j = min(i + 22, n)
        preds.append(pd.Series(lr.predict(X.iloc[i:j]), index=y.index[i:j]))
    return np.exp(pd.concat(preds))


def stats(ret: pd.Series, label: str):
    eq = ret.cumsum()
    dd = (eq - eq.cummax()).min()
    print(f"{label:26s} ann.ret {ret.mean()*252*100:6.1f}%  ann.vol {ret.std()*np.sqrt(252)*100:5.1f}%  "
          f"Sharpe {ret.mean()/ret.std()*np.sqrt(252):5.2f}  maxDD {dd*100:6.1f}%  t={nw_tstat(ret.to_numpy(),0):.2f}")


def main():
    df = load_mid_with_spread("H1"); df = df[df.index >= "2017-01-01"]
    r = np.log(df["close"]).diff()
    day = r.index.normalize()
    rv = (r ** 2).groupby(day).sum().pow(0.5); rv = rv[rv > 0]
    # daily close-to-close return and spread, aligned to days
    dclose = df["close"].groupby(day).last(); dret = np.log(dclose).diff()
    dspread = (df["spread"] / df["close"]).groupby(day).median()
    fc = har_forecast(rv)
    idx = fc.index.intersection(dret.index)
    fc, dret, dspread = fc[idx], dret[idx], dspread[idx]
    target = float(rv[:fc.index[0]].mean())  # target = historical average daily vol
    print(f"OOS {idx[0].date()}..{idx[-1].date()} n={len(idx)}  target daily vol {target*1e4:.0f} bps")
    results = {}
    const = dret - 0.0
    results["constant long 1x"] = const
    for cap in (1.5, 2.0):
        w = (target / fc).clip(upper=cap)
        w = w.shift(1).fillna(1.0)  # forecast made at prior close -> position for today
        turn = w.diff().abs().fillna(w.iloc[0])
        results[f"vol-target (cap {cap}x)"] = w * dret - turn * dspread
        print(f"cap {cap}: mean exposure {w.mean():.2f}, range {w.min():.2f}-{w.max():.2f}, avg daily turnover {turn.mean():.3f}")
    # naive: yesterday's rv as forecast (is HAR adding anything over the trivial rule?)
    w_naive = (target / rv.reindex(idx).shift(1)).clip(upper=2.0).fillna(1.0)
    results["vol-target naive(yday rv)"] = w_naive * dret - w_naive.diff().abs().fillna(0) * dspread
    print()
    for k, v in results.items(): stats(v.dropna(), k)
    print("\nper-year Sharpe:")
    tbl = pd.DataFrame({k: v.groupby(v.index.year).apply(lambda x: x.mean()/x.std()*np.sqrt(252)) for k, v in results.items()})
    print(tbl.round(2).to_string())


if __name__ == "__main__":
    main()
