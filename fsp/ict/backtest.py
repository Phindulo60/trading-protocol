"""Walk-forward backtester for the ICT confluence engine.

Reuses the realistic execution machinery from ``fsp.backtest.engine`` (limit
fills, SL slippage, spread, max-hold timeout, cooldown) and swaps the signal
source to ``fsp.ict.engine.decide``. This keeps the ICT results directly
comparable to the legacy grader backtest.

Causality: at each LTF bar i we only ever pass ``ltf_df.iloc[i-window:i+1]`` to
the engine, and the HTF frame is sliced to bars closed at/before that bar. No
look-ahead. One position at a time per pair (limit entry, single TP).
"""
from __future__ import annotations

from typing import Callable

import pandas as pd

from fsp.backtest.engine import (
    Trade, BacktestResult, ExecConfig,
    _check_fill, _update_open, _realised_r,
)
from fsp.ict.engine import decide, TradeDecision
from fsp.ict.smt import partner_for
from fsp.context.sessions import session_of
from fsp.data.feed import default_feed

GRADE_RANK = {"B": 1, "A": 2, "A+": 3}

# A decider takes the LTF window + (optional) HTF window and returns a decision.
Decider = Callable[..., TradeDecision]


def simulate_ict(
    ltf_df: pd.DataFrame,
    pair: str,
    htf_df: pd.DataFrame | None = None,
    *,
    decider: Decider = decide,
    min_grade: str = "A",
    window: int = 350,
    swing_length: int = 5,
    lookback: int = 30,
    atr_mult: float = 1.5,
    atr_len: int = 20,
    stride: int = 1,
    htf_window: int = 200,
    htf_drop_forming: bool = True,
    tp_cap_r: float | None = None,
    exec_cfg: ExecConfig | None = None,
    tz: str = "America/New_York",
    smt_df: pd.DataFrame | None = None,
    smt_sign: int = 1,
    smt_partner: str | None = None,
    context_window: int = 1500,
) -> BacktestResult:
    """Replay `ltf_df` bar-by-bar; open at most one ICT trade at a time."""
    cfg = exec_cfg or ExecConfig(partial_pct=1.0, min_rr_tp1=1.5)
    pip = 0.01 if "JPY" in pair else 0.0001
    min_rank = GRADE_RANK.get(min_grade, 2)

    result = BacktestResult(
        start=ltf_df.index[0].to_pydatetime() if len(ltf_df) else None,
        end=ltf_df.index[-1].to_pydatetime() if len(ltf_df) else None,
    )
    open_trade: Trade | None = None
    last_signal_bar = -10**9
    last_dedup: str | None = None
    skip: dict[str, int] = {}

    def bump(r):
        skip[r] = skip.get(r, 0) + 1

    n = len(ltf_df)
    for i in range(0, n, max(1, stride)):
        ts = ltf_df.index[i]

        # ---- manage an existing position ----
        if open_trade is not None:
            bar = ltf_df.iloc[i]
            if open_trade.outcome == "pending":
                _check_fill(open_trade, bar, ltf_df, i, cfg, pip)
            if open_trade.outcome == "open":
                _update_open(open_trade, bar, ltf_df, i, cfg, pip)
            if open_trade.outcome == "pending":
                open_trade.pending_bars += 1
                if open_trade.pending_bars >= cfg.max_pending_bars:
                    open_trade.outcome = "be"
                    open_trade.r_multiple = 0.0
                    open_trade.weighted_r = 0.0
            if open_trade.outcome not in ("pending", "open"):
                # drop never-filled cancellations so they don't pollute stats
                if not (open_trade.outcome == "be" and open_trade.filled_ts is None):
                    result.trades.append(open_trade)
                open_trade = None

        # ---- look for a new signal ----
        if open_trade is None and i >= window and (i - last_signal_bar) >= cfg.cooldown_bars:
            win = ltf_df.iloc[i - window:i + 1]
            ctx = ltf_df.iloc[max(0, i - context_window):i + 1]   # wider ctx for PDH/PDL/PWH
            hwin = None
            if htf_df is not None:
                pos = htf_df.index.searchsorted(ts, side="right")  # bars[0:pos] have index <= ts
                end = max(0, pos - 1) if htf_drop_forming else pos  # drop forming HTF bar
                hwin = htf_df.iloc[max(0, end - htf_window):end]
            try:
                d = decider(win, hwin, pair=pair, swing_length=swing_length,
                            lookback=lookback, atr_mult=atr_mult, atr_len=atr_len, tz=tz,
                            smt_df=smt_df, smt_sign=smt_sign, smt_partner=smt_partner,
                            context_df=ctx)
            except Exception:
                bump("decider-error"); continue

            if not d.is_tradable:
                bump("not-tradable"); continue
            if GRADE_RANK.get(d.grade, 0) < min_rank:
                bump("grade"); continue
            if d.rr is None or d.rr < cfg.min_rr_tp1:
                bump("rr<min"); continue

            dk = f"{d.grade}|{d.direction}|{d.entry:.5f}|{d.stop:.5f}"
            if dk == last_dedup:
                bump("dedup"); continue
            last_dedup = dk
            last_signal_bar = i

            entry, stop, target, rr = d.entry, d.stop, d.target, d.rr
            if tp_cap_r is not None and rr is not None and rr > tp_cap_r:
                risk = abs(entry - stop)
                target = entry + tp_cap_r * risk if d.direction == "long" else entry - tp_cap_r * risk
                rr = tp_cap_r
            local = ts.tz_convert(tz)
            spread = cfg.spread_pips * pip
            fill = entry + spread / 2 if d.direction == "long" else entry - spread / 2
            open_trade = Trade(
                open_ts=ts.to_pydatetime(), close_ts=None, pair=pair,
                direction=d.direction, grade=d.grade,
                entry=entry, fill=fill, sl=stop, tp1=target, tp2=None,
                risk_r=1.0, rr_tp1=rr, rr_tp2=None,
                checklist_passed=d.score, checklist_total=12,
                session=session_of(ts, tz).value, dow=local.weekday(),
                smt=d.smt, target_kind=d.target_kind, sweep_major=d.sweep_major,
            )

    # close a still-open trade at the last bar (excluded from win/loss stats)
    if open_trade is not None and open_trade.outcome == "open":
        last_bar = ltf_df.iloc[-1]
        r = _realised_r(open_trade.fill, float(last_bar["close"]), open_trade.sl,
                        open_trade.direction, pip, cfg)
        open_trade.outcome = "eop"
        open_trade.r_multiple = r
        open_trade.weighted_r = r * open_trade.risk_r
        open_trade.exit_price = float(last_bar["close"])
        open_trade.close_ts = ltf_df.index[-1].to_pydatetime()
        result.trades.append(open_trade)

    result.skip_reasons = skip
    return result


def run_ict_backtest(
    pair: str,
    start,
    end,
    ltf: str = "M15",
    htf: str | None = "H1",
    feed_kind: str = "duka",
    **kw,
) -> BacktestResult:
    """Fetch history via the data feed and run a single-pair ICT backtest."""
    f = default_feed(feed_kind)
    ltf_df = f.history(pair, ltf, start, end)
    htf_df = f.history(pair, htf, start, end) if htf else None
    # load SMT partner (same timeframe) if a correlation pairing is defined
    smt_kw: dict = {}
    pt = partner_for(pair)
    if pt is not None:
        p_pair, p_sign = pt
        try:
            smt_kw = dict(smt_df=f.history(p_pair, ltf, start, end),
                          smt_sign=p_sign, smt_partner=p_pair)
        except Exception:
            pass  # partner unavailable — proceed without SMT
    return simulate_ict(ltf_df, pair, htf_df, **smt_kw, **kw)


def aggregate(results: dict[str, BacktestResult]) -> BacktestResult:
    """Merge per-pair results into one for combined stats."""
    merged = BacktestResult()
    for r in results.values():
        merged.trades.extend(r.trades)
    starts = [r.start for r in results.values() if r.start]
    ends = [r.end for r in results.values() if r.end]
    merged.start = min(starts) if starts else None
    merged.end = max(ends) if ends else None
    return merged


def report(results: dict[str, BacktestResult]) -> str:
    """Human-readable per-pair + aggregate summary."""
    lines = []
    hdr = f"{'pair':<8}{'n':>5}{'WR':>8}{'totR':>9}{'exp':>8}{'PF':>7}{'maxDD':>9}"
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for pair, r in sorted(results.items()):
        s = r.stats()
        if s.get("total", 0) == 0:
            lines.append(f"{pair:<8}{0:>5}{'-':>8}{'-':>9}{'-':>8}{'-':>7}{'-':>9}")
            continue
        pf = s["profit_factor"]
        pf_s = "inf" if pf == float("inf") else f"{pf:.2f}"
        lines.append(f"{pair:<8}{s['total']:>5}{s['win_rate']*100:>7.1f}%"
                     f"{s['total_r']:>9.1f}{s['expectancy']:>8.2f}{pf_s:>7}{s['max_dd']:>9.1f}")
    agg = aggregate(results).stats()
    if agg.get("total", 0):
        lines.append("-" * len(hdr))
        pf = agg["profit_factor"]
        pf_s = "inf" if pf == float("inf") else f"{pf:.2f}"
        lines.append(f"{'ALL':<8}{agg['total']:>5}{agg['win_rate']*100:>7.1f}%"
                     f"{agg['total_r']:>9.1f}{agg['expectancy']:>8.2f}{pf_s:>7}{agg['max_dd']:>9.1f}")
        gd = agg.get("by_grade", {})
        lines.append("")
        lines.append("by grade: " + "  ".join(
            f"{g}: n={d['n']} wr={d['wr']*100:.0f}% exp={d['exp']:.2f}"
            for g, d in sorted(gd.items())))
    return "\n".join(lines)
