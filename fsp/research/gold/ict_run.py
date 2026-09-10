"""ICT confluence engine on XAUUSD: parameterised backtest with gold-correct costs.

Uses fsp.ict.backtest.simulate_ict (limit fills, SL slippage, one position at a time).
Gold cost: spread 30 points ($0.30/oz, TradeNation ~0.28), SL slippage 20 points.

Gold-specific fixes over the FX defaults:
  --sl-buffer   ATR multiple added beyond the swept extreme (FX default 0.1 is inside gold noise)
  --min-risk    reject setups whose |entry-stop| < this many ATR (keeps spread <~10% of risk)
  --ltf/--htf   H1/H4 structure as an alternative to M15/H1
"""
from __future__ import annotations

import argparse, time
from functools import partial
from pathlib import Path
import numpy as np
import pandas as pd

from fsp.research.gold.data import load_mid_with_spread
from fsp.ict.backtest import simulate_ict, report
from fsp.ict.engine import decide, TradeDecision
from fsp.backtest.engine import ExecConfig
from fsp.structure.displacement import atr as _atr

OUT = Path.home() / ".fsp" / "cache" / "gold" / "ict"


def resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    o = df.resample(rule, label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
    return o.dropna(subset=["open"])


def frames(ltf: str, htf: str, start: str, end: str):
    base = load_mid_with_spread("M15" if ltf in ("M15",) else "H1")
    cols = ["open", "high", "low", "close", "volume"]
    s0 = pd.Timestamp(start, tz="UTC") - pd.Timedelta(days=60)
    base = base[(base.index >= s0) & (base.index < end)][cols]
    l = base if ltf in ("M15", "H1") else resample(base, ltf)
    if htf == "H1":
        h = load_mid_with_spread("H1"); h = h[(h.index >= s0) & (h.index < end)][cols]
    elif htf == "H4":
        h1 = load_mid_with_spread("H1"); h1 = h1[(h1.index >= s0) & (h1.index < end)][cols]
        h = resample(h1, "4h")
    else:
        raise ValueError(htf)
    l = l[l.index >= start]
    return l, h


def make_decider(sl_buffer_atr: float, min_risk_atr: float, atr_len: int = 20):
    base = partial(decide, sl_buffer_atr=sl_buffer_atr)

    def dec(win, hwin, **kw) -> TradeDecision:
        d = base(win, hwin, **kw)
        if d.entry is not None and d.stop is not None and min_risk_atr > 0:
            a = float(_atr(win, atr_len).iloc[-1])
            if abs(d.entry - d.stop) < min_risk_atr * a:
                d.grade = "skip"; d.missing.append(f"risk<{min_risk_atr}ATR")
        return d
    return dec


def run(args) -> pd.DataFrame:
    l, h = frames(args.ltf, args.htf, args.start, args.end)
    tf_min = {"M15": 15, "H1": 60}[args.ltf]
    cfg = ExecConfig(spread_pips=args.spread, sl_slippage_pips=20, partial_pct=1.0, min_rr_tp1=args.min_rr,
                     max_hold_bars=int(args.max_hold_h * 60 / tf_min), cooldown_bars=2,
                     max_pending_bars=int(args.pending_h * 60 / tf_min),
                     skip_lunch=False, skip_monday_before_ny=False, skip_same_session_after_loss=False)
    t = time.time()
    res = simulate_ict(l, "XAUUSD", h, decider=make_decider(args.sl_buffer, args.min_risk),
                       min_grade=args.min_grade, exec_cfg=cfg, tp_cap_r=args.tp_cap, stride=args.stride,
                       window=args.window, htf_window=args.htf_window)
    print(f"[{args.tag}] ltf={args.ltf} htf={args.htf} bars={len(l)} {time.time()-t:.0f}s skips={res.skip_reasons}")
    print(report({"XAUUSD": res}))
    rows = [{k: getattr(tr, k) for k in ("open_ts", "filled_ts", "close_ts", "direction", "grade", "entry", "fill", "sl", "tp1",
             "rr_tp1", "outcome", "r_multiple", "session", "dow", "checklist_passed", "target_kind", "exit_price")} for tr in res.trades]
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True); df.to_parquet(OUT / f"trades_{args.tag}.parquet")
    if len(df):
        r = df.r_multiple.to_numpy()
        print(f"expectancy {r.mean():+.3f}R t={r.mean()/r.std()*np.sqrt(len(r)):.2f} n={len(r)}  "
              f"median risk $/oz {(df.fill-df.sl).abs().median():.2f}  outcomes {df.outcome.value_counts().to_dict()}")
        print("by year:", df.groupby(pd.to_datetime(df.open_ts).dt.year).r_multiple.agg(["count", "mean"]).round(2).to_dict("index"))
        print("by grade:", df.groupby("grade").r_multiple.agg(["count", "mean"]).round(2).to_dict("index"))
        print("by session:", df.groupby("session").r_multiple.agg(["count", "mean"]).round(2).to_dict("index"))
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2022-02-01"); ap.add_argument("--end", default="2026-09-01")
    ap.add_argument("--ltf", default="M15"); ap.add_argument("--htf", default="H1")
    ap.add_argument("--min-grade", default="B"); ap.add_argument("--spread", type=float, default=30)
    ap.add_argument("--tp-cap", type=float, default=None); ap.add_argument("--min-rr", type=float, default=1.5)
    ap.add_argument("--sl-buffer", type=float, default=0.1); ap.add_argument("--min-risk", type=float, default=0.0)
    ap.add_argument("--max-hold-h", type=float, default=16); ap.add_argument("--pending-h", type=float, default=3)
    ap.add_argument("--window", type=int, default=350); ap.add_argument("--htf-window", type=int, default=200)
    ap.add_argument("--stride", type=int, default=1); ap.add_argument("--tag", default="base")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
