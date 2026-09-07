"""Experiment 2: daily gold, horizons 1/5/20 days, with lagged macro drivers.

Usage: .venv-ml/bin/python -m fsp.research.gold.run_daily --h 5
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
BARS_PER_YEAR_D = 252


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h", type=int, default=5)
    ap.add_argument("--no-macro", action="store_true")
    ap.add_argument("--start", default="2011-01-01")
    ap.add_argument("--folds", type=int, default=6)
    args = ap.parse_args()

    df = load_mid_with_spread("D")
    df = df[df.index >= args.start]
    # Dukascopy daily bars carry a 22:00 UTC stamp for the prior-day session; normalise to date
    df.index = df.index.normalize()
    df = df[~df.index.duplicated(keep="last")]
    f = build_features(df, bars_per_day=1)
    f = f.drop(columns=["hour", "hour_sin", "hour_cos", "is_london", "is_ny", "is_asia", "dom"])
    if not args.no_macro:
        macro = load_macro()
        gs = np.log(df["close"] / macro["silver"].reindex(df.index, method="ffill"))
        f = attach_macro(f, macro)
        f["gold_silver_z60"] = ((gs - gs.rolling(60).mean()) / gs.rolling(60).std()).shift(1)
        # gold vs real-yield proxy: 20d beta residual (gold "too high/low" for TIP)
        tip = np.log(macro["tip"].reindex(df.index, method="ffill")).shift(1)
        gl = np.log(df["close"])
        cov = gl.diff().rolling(60).cov(tip.diff()); var = tip.diff().rolling(60).var()
        f["gold_tip_resid20"] = (gl.diff(20) - (cov / var) * tip.diff(20)).shift(0) / f["rv_20d"]
    y = forward_return(df, args.h)
    cost = (df["spread"] / df["close"]).rolling(5).median()
    X = f.astype(float)
    print(f"rows={len(X)} features={X.shape[1]} h={args.h} start={X.index[0].date()} end={X.index[-1].date()}")
    print(f"median spread bps={float(cost.median()*1e4):.2f}  label sd bps={float(y.std()*1e4):.1f}")

    q50 = lambda p: float(np.quantile(np.abs(p), 0.5))
    lg_small = dict(n_estimators=200, num_leaves=7, max_depth=3, min_child_samples=60)
    models = [
        ("zero", lambda: ZeroForecaster(), 0.0),
        ("drift", lambda: MeanForecaster(), 0.0),
        ("tsmom(ret_168)", lambda: MomentumForecaster("ret_168"), 0.0),  # 168 bars = ~8 months on D
        ("tsmom(ret_24)", lambda: MomentumForecaster("ret_24"), 0.0),
        ("ridge", lambda: RidgeForecaster(alpha=300.0), 0.0),
        ("ridge@q50", lambda: RidgeForecaster(alpha=300.0), q50),
        ("lgbm_reg", lambda: LGBMForecaster(**lg_small), 0.0),
        ("lgbm_reg@q50", lambda: LGBMForecaster(**lg_small), q50),
        ("lgbm_clf@q50", lambda: LGBMClassifierForecaster(**lg_small), q50),
    ]
    rows, results = [], {}
    for name, mk, thr in models:
        r = walk_forward(name, mk, X, y, cost, h=args.h, bars_per_year=BARS_PER_YEAR_D,
                         n_folds=args.folds, threshold=thr)
        rows.append(r.summary); results[name] = r
    out = pd.DataFrame(rows).set_index("name")
    pd.set_option("display.width", 220)
    print(out.round(4).to_string())
    best = out["excess_t_nw"].idxmax()
    print(f"\nper-fold: {best}")
    print(results[best].table().round(3).to_string(index=False))
    lg = LGBMForecaster(**lg_small); m = X.notna().all(axis=1) & y.notna()
    lg.fit(X[m].iloc[:int(m.sum()*0.8)], y[m].iloc[:int(m.sum()*0.8)])
    imp = pd.Series(lg.importances, index=X.columns).sort_values(ascending=False)
    print("\nlgbm top features:", imp.head(12).to_dict())


if __name__ == "__main__":
    main()
