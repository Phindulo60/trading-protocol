"""Structure checks that a generic return model cannot express.

1. Hour-of-day seasonality: mean H1 return per UTC hour with Newey-West t,
   estimated on the first half and re-tested on the second half.
2. Volatility forecastability: HAR-RV (Corsi) for next-24h realised vol vs
   the naive 'yesterday's vol' forecast, out-of-sample R^2 and QLIKE.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fsp.research.gold.data import load_mid_with_spread
from fsp.research.gold.evaluate import nw_tstat


def hour_seasonality():
    df = load_mid_with_spread("H1")
    df = df[df.index >= "2017-01-01"]
    r = np.log(df["close"]).diff().dropna()
    cost_bps = float((df["spread"] / df["close"]).median() * 1e4)
    mid = r.index[len(r) // 2]
    a, b = r[r.index < mid], r[r.index >= mid]
    rows = []
    for hr in range(24):
        ra, rb = a[a.index.hour == hr], b[b.index.hour == hr]
        rows.append({"hour": hr, "n": len(ra) + len(rb),
                     "mean_bps_1st": ra.mean() * 1e4, "t_1st": nw_tstat(ra.to_numpy(), 0),
                     "mean_bps_2nd": rb.mean() * 1e4, "t_2nd": nw_tstat(rb.to_numpy(), 0),
                     "sd_bps": r[r.index.hour == hr].std() * 1e4})
    out = pd.DataFrame(rows).set_index("hour")
    print(f"H1 hour-of-day (UTC), split {mid.date()}, round-trip spread {cost_bps:.2f} bps")
    print(out.round(2).to_string())
    sig = out[(out["t_1st"].abs() > 2.5) & (np.sign(out["t_1st"]) == np.sign(out["t_2nd"])) & (out["t_2nd"].abs() > 2)]
    print("\nhours significant in 1st half AND same-sign significant in 2nd:", sig.index.tolist())
    return out


def har_rv():
    df = load_mid_with_spread("H1")
    df = df[df.index >= "2017-01-01"]
    r = np.log(df["close"]).diff()
    day = r.index.normalize()
    rv = (r ** 2).groupby(day).sum().pow(0.5)  # daily realised vol from hourly returns
    rv = rv[rv > 0]
    lrv = np.log(rv)
    X = pd.DataFrame({"d": lrv.shift(1), "w": lrv.rolling(5).mean().shift(1),
                      "m": lrv.rolling(22).mean().shift(1)})
    y = lrv
    m = X.notna().all(axis=1) & y.notna()
    X, y = X[m], y[m]
    n = len(y); cut = int(n * 0.5)
    from sklearn.linear_model import LinearRegression
    preds, naive = [], []
    # expanding walk-forward, refit monthly
    for i in range(cut, n, 22):
        lr = LinearRegression().fit(X.iloc[:i], y.iloc[:i])
        j = min(i + 22, n)
        preds.append(pd.Series(lr.predict(X.iloc[i:j]), index=y.index[i:j]))
        naive.append(X["d"].iloc[i:j])
    p, nv, yt = pd.concat(preds), pd.concat(naive), y.iloc[cut:]
    def r2(pred): return 1 - ((yt - pred) ** 2).sum() / ((yt - yt.mean()) ** 2).sum()
    def qlike(pred):
        s2, f2 = np.exp(2 * yt), np.exp(2 * pred)
        return float((s2 / f2 - np.log(s2 / f2) - 1).mean())
    print(f"\nHAR-RV next-day vol, OOS {yt.index[0].date()}..{yt.index[-1].date()} n={len(yt)}")
    print(f"  R2 (log rv): HAR={r2(p):.3f}  naive(yesterday)={r2(nv):.3f}  mean={0:.3f}")
    print(f"  QLIKE:       HAR={qlike(p):.4f}  naive={qlike(nv):.4f}  (lower better)")
    print(f"  coefs (last fit): {dict(zip(X.columns, lr.coef_.round(3)))}")
    # does predicted vol rank actually order realised vol? decile spread
    q = pd.qcut(p, 5, labels=False)
    print("  realised vol (bps/day) by forecast quintile:", (np.exp(yt).groupby(q).mean() * 1e4).round(0).tolist())


if __name__ == "__main__":
    hour_seasonality()
    har_rv()
