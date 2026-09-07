"""Experiment 1: H1 gold, horizons 4/24 bars, models vs baselines.

Usage: .venv-ml/bin/python -m fsp.research.gold.run_h1 [--h 24] [--macro]
"""
from __future__ import annotations

import argparse
import logging
import numpy as np
import pandas as pd

from fsp.research.gold.data import load_mid_with_spread, load_macro
from fsp.research.gold.features import build_features, attach_macro, forward_return
from fsp.research.gold.evaluate import (walk_forward, ZeroForecaster, MeanForecaster,
                                        MomentumForecaster, RidgeForecaster,
                                        LGBMForecaster, LGBMClassifierForecaster)

logging.basicConfig(level=logging.WARNING)
BARS_PER_YEAR_H1 = 24 * 5 * 52 * 0.96  # ~5990 traded hours


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h", type=int, default=24)
    ap.add_argument("--macro", action="store_true")
    ap.add_argument("--start", default="2017-01-01")
    ap.add_argument("--folds", type=int, default=6)
    args = ap.parse_args()

    df = load_mid_with_spread("H1")
    df = df[df.index >= args.start]
    f = build_features(df, bars_per_day=24)
    if args.macro:
        f = attach_macro(f, load_macro())
    y = forward_return(df, args.h)
    # round-trip cost in log-return units: spread/price, observed at entry bar
    cost = (df["spread"] / df["close"]).rolling(24).median()
    X = f.drop(columns=["hour", "dow", "dom"]).astype(float)
    print(f"rows={len(X)} features={X.shape[1]} h={args.h} start={X.index[0].date()} end={X.index[-1].date()}")
    print(f"median spread bps={float(cost.median()*1e4):.2f}  label sd bps={float(y.std()*1e4):.1f}")

    mom_col = "ret_168" if args.h >= 24 else "ret_24"
    q50 = lambda p: float(np.quantile(np.abs(p), 0.5))
    models = [
        ("zero", lambda: ZeroForecaster(), 0.0),
        ("drift", lambda: MeanForecaster(), 0.0),
        (f"tsmom({mom_col})", lambda: MomentumForecaster(mom_col), 0.0),
        ("ridge", lambda: RidgeForecaster(alpha=100.0), 0.0),
        ("ridge@q50", lambda: RidgeForecaster(alpha=100.0), q50),
        ("lgbm_reg", lambda: LGBMForecaster(), 0.0),
        ("lgbm_reg@q50", lambda: LGBMForecaster(), q50),
        ("lgbm_clf@q50", lambda: LGBMClassifierForecaster(), q50),
    ]
    rows = []
    results = {}
    for name, mk, thr in models:
        r = walk_forward(name, mk, X, y, cost, h=args.h, bars_per_year=BARS_PER_YEAR_H1,
                         n_folds=args.folds, threshold=thr)
        rows.append(r.summary)
        results[name] = r
    out = pd.DataFrame(rows).set_index("name")
    pd.set_option("display.width", 200)
    print(out.round(4).to_string())

    # per-fold stability for the best model by t-stat
    best = out["tstat_nw"].idxmax()
    print(f"\nper-fold: {best}")
    print(results[best].table().round(3).to_string(index=False))

    lg = LGBMForecaster(); m = X.notna().all(axis=1) & y.notna()
    lg.fit(X[m].iloc[:int(m.sum()*0.8)], y[m].iloc[:int(m.sum()*0.8)])
    imp = pd.Series(lg.importances, index=X.columns).sort_values(ascending=False)
    print("\nlgbm top features:", imp.head(12).to_dict())


if __name__ == "__main__":
    main()
