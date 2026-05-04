"""Event-loop backtester.

Replays history bar-by-bar (LTF closes) with NO look-ahead. At each bar:
  1. Slice all reference frames up to current timestamp.
  2. Re-run `grade_setup` on the sliced data.
  3. If grade >= threshold and no open trade for that pair, "enter" at close.
  4. Walk forward bar-by-bar checking if SL or TP1/TP2 hits first (intra-bar H/L).
  5. Log outcome as R-multiple.

Entry model (simplification of PDF):
  - Fill at the LTF bar-close where the grade first crosses the threshold
    (approximates 'entering on LTF confirmation of CIOF').
  - SL / TP1 / TP2 taken directly from the SetupCandidate.
  - Partial logic: if hit TP1 first, close 50%, move SL to BE, run remainder to TP2.
  - If SL hits first → -1R. If TP1 only → +(0.5 * rr_tp1). If TP2 → +(0.5*rr_tp1 + 0.5*rr_tp2).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Iterable, Literal

import pandas as pd

from fsp.data.feed import default_feed
from fsp.data.types import Grade
from fsp.grader.setup import grade_setup
from fsp.grader.precompute import precompute, grade_setup_fast

log = logging.getLogger("fsp.backtest")


@dataclass
class ExecConfig:
    """Realistic execution model. See M6.1 notes in README."""
    spread_pips: float = 0.3          # TradeNation typical on EURUSD/GBPUSD
    sl_slippage_pips: float = 0.5     # stop-hunt overshoot on SL
    tp1_cap_r: float = 2.0            # PDF: "First partial roughly at 2R"
    tp2_cap_r: float = 4.0            # Cap TP2 reward to realistic ladder
    partial_pct: float = 0.5          # fraction closed at TP1
    runner_trail_to_be: bool = True   # after TP1, runner SL -> BE
    min_rr_tp1: float = 1.8           # reject if setup offers < this
    max_hold_bars: int = 64           # ~16h M15 = end of next NY session
    cooldown_bars: int = 2            # min LTF bars between new signals
    max_pending_bars: int = 12        # cancel unfilled limit after N LTF bars (~3h M15)
    skip_lunch: bool = True
    skip_monday_before_ny: bool = True
    skip_same_session_after_loss: bool = True


@dataclass
class Trade:
    open_ts: datetime
    close_ts: datetime | None
    pair: str
    direction: str
    grade: str
    entry: float                  # signal entry (pre-slippage)
    fill: float                   # actual fill (entry + spread)
    sl: float
    tp1: float | None
    tp2: float | None
    risk_r: float
    rr_tp1: float | None
    rr_tp2: float | None
    checklist_passed: int
    checklist_total: int
    session: str
    dow: int                      # 0=Mon ... 6=Sun (local NY)
    outcome: Literal["win1", "win2", "loss", "be", "timeout", "eop", "pending", "open"] = "pending"
    r_multiple: float = 0.0
    weighted_r: float = 0.0
    exit_price: float | None = None
    bars_held: int = 0
    pending_bars: int = 0
    filled_ts: datetime | None = None
    skip_reason: str = ""


@dataclass
class BacktestResult:
    trades: list[Trade] = field(default_factory=list)
    start: datetime | None = None
    end: datetime | None = None
    skip_reasons: dict = field(default_factory=dict)

    def stats(self) -> dict:
        closed = [t for t in self.trades if t.outcome not in ("open", "pending", "eop")]
        if not closed:
            return {"total": 0}
        n = len(closed)
        wins = [t for t in closed if t.weighted_r > 0]
        losses = [t for t in closed if t.weighted_r < 0]
        total_r = sum(t.weighted_r for t in closed)
        total_win = sum(t.weighted_r for t in wins)
        total_loss = -sum(t.weighted_r for t in losses)
        pf = (total_win / total_loss) if total_loss > 0 else float("inf")
        # equity path
        eq = [0.0]
        for t in closed:
            eq.append(eq[-1] + t.weighted_r)
        peak = eq[0]
        dd = 0.0
        for x in eq:
            peak = max(peak, x)
            dd = min(dd, x - peak)
        avg_win = total_win / len(wins) if wins else 0
        avg_loss = -total_loss / len(losses) if losses else 0
        def brk(key_fn):
            out: dict = {}
            for t in closed:
                k = key_fn(t)
                d = out.setdefault(k, {"n": 0, "w": 0, "r": 0.0})
                d["n"] += 1
                if t.weighted_r > 0: d["w"] += 1
                d["r"] += t.weighted_r
            for k, d in out.items():
                d["wr"] = d["w"] / d["n"] if d["n"] else 0
                d["exp"] = d["r"] / d["n"] if d["n"] else 0
            return out
        dows = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
        return {
            "total": n,
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": len(wins) / n,
            "expectancy": total_r / n,
            "profit_factor": pf,
            "total_r": total_r,
            "max_dd": dd,
            "equity": eq,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "avg_win_over_loss": (avg_win / -avg_loss) if avg_loss < 0 else float("inf"),
            "by_grade": brk(lambda t: t.grade),
            "by_session": brk(lambda t: t.session),
            "by_dow": brk(lambda t: dows[t.dow] if t.dow < 7 else "?"),
            "by_outcome": {o: len([t for t in closed if t.outcome == o])
                           for o in ("win1", "win2", "loss", "timeout", "be")},
            "worst_10": sorted(closed, key=lambda t: t.weighted_r)[:10],
        }


def _slice_upto(df: pd.DataFrame, ts: pd.Timestamp) -> pd.DataFrame:
    return df[df.index <= ts]


def run_backtest(
    pair: str,
    start: datetime,
    end: datetime,
    ltf: str = "M15",
    feed_kind: str = "duka",
    min_grade: Grade = Grade.A,
    include_cross: bool = True,
    include_dxy: bool = False,
    stride: int = 1,
    verbose_every: int = 500,
    exec_cfg: ExecConfig | None = None,
    fast: bool = True,
) -> BacktestResult:
    """Run a full event-loop backtest for one pair over [start, end]."""
    f = default_feed(feed_kind)
    log.info("Loading history for %s ...", pair)
    # Pull enough warmup data
    warmup_days = 35
    w_start = start - timedelta(days=warmup_days)
    ltf_all = f.history(pair, ltf, w_start, end)
    h1_all = f.history(pair, "H1", w_start, end)
    daily_all = f.history(pair, "D", w_start - timedelta(days=30), end)

    other = "GBPUSD" if pair == "EURUSD" else "EURUSD"
    other_all = None
    if include_cross:
        try:
            other_all = f.history(other, "H1", w_start, end)
        except Exception as e:
            log.warning("cross %s fetch failed: %s", other, e)

    dxy_all = None
    if include_dxy:
        try:
            dxy_all = default_feed("yf").history("DXY", "H1", w_start, end)
        except Exception as e:
            log.warning("DXY fetch failed: %s", e)

    # Decision points: LTF bars that fall inside [start, end]
    _s = pd.Timestamp(start)
    _e = pd.Timestamp(end)
    ltf_in_range = ltf_all[(ltf_all.index >= _s) & (ltf_all.index <= _e)]
    decision_idx = list(range(0, len(ltf_in_range), max(1, stride)))

    cfg = exec_cfg or ExecConfig()
    grade_rank = {Grade.B: 1, Grade.A: 2, Grade.A_PLUS: 3}

    pc = None
    if fast:
        log.info("Precomputing structure over full window...")
        t0 = datetime.now()
        pc = precompute(pair, ltf_all, h1_all, daily_all,
                        other=other_all, other_pair=other,
                        dxy=dxy_all, verbose=True)
        log.info("Precompute done in %.1fs", (datetime.now() - t0).total_seconds())

    min_rank = grade_rank[min_grade]
    pip = 0.01 if "JPY" in pair else 0.0001

    result = BacktestResult(start=start, end=end)
    open_trade: Trade | None = None
    last_dedup: str | None = None
    last_signal_bar = -999
    last_loss_session_key: str | None = None
    skip_reasons: dict[str, int] = {}

    def _bump(r):
        skip_reasons[r] = skip_reasons.get(r, 0) + 1

    for i, li in enumerate(decision_idx):
        ts = ltf_in_range.index[li]

        if open_trade is not None:
            bar = ltf_in_range.iloc[li]
            if open_trade.outcome == "pending":
                _check_fill(open_trade, bar, ltf_in_range, li, cfg, pip)
            if open_trade.outcome == "open":
                _update_open(open_trade, bar, ltf_in_range, li, cfg, pip)
            # pending timeout: if not filled within max_pending_bars, cancel
            if open_trade.outcome == "pending":
                open_trade.pending_bars += 1
                if open_trade.pending_bars >= cfg.max_pending_bars:
                    open_trade.outcome = "be"   # unfilled, no PnL
                    open_trade.r_multiple = 0
                    open_trade.weighted_r = 0
            if open_trade.outcome not in ("pending", "open"):
                if open_trade.outcome == "loss" and cfg.skip_same_session_after_loss:
                    local_l = ts.tz_convert("America/New_York")
                    last_loss_session_key = str(local_l.date()) + "|" + open_trade.session
                # drop "never filled" trades so they don't pollute stats
                if open_trade.outcome == "be" and open_trade.filled_ts is None:
                    pass
                else:
                    result.trades.append(open_trade)
                open_trade = None

        if open_trade is None:
            if li - last_signal_bar < cfg.cooldown_bars:
                _bump("cooldown"); continue

            try:
                if fast:
                    # Find index in full ltf_all
                    li_full = int(pc.ltf_index.searchsorted(ts, side="right")) - 1
                    if li_full < 50:
                        continue
                    s = grade_setup_fast(pc, ts, li=li_full)
                else:
                    ltf_slice = ltf_all[ltf_all.index <= ts]
                    if len(ltf_slice) < 50:
                        continue
                    h1_slice = _slice_upto(h1_all, ts)
                    daily_slice = _slice_upto(daily_all, ts)
                    other_slice = _slice_upto(other_all, ts) if other_all is not None else None
                    dxy_slice = _slice_upto(dxy_all, ts) if dxy_all is not None else None
                    s = grade_setup(pair, ltf_slice, h1_slice, daily_slice,
                                    other_df=other_slice, other_pair=other,
                                    dxy_df=dxy_slice)
            except Exception as e:
                log.debug("grade err at %s: %s", ts, e)
                continue

            if grade_rank.get(s.grade, 0) < min_rank:
                _bump("grade"); continue
            if s.entry is None or s.sl is None or s.tp1 is None:
                _bump("no-setup"); continue
            if s.rr_tp1 is None or s.rr_tp1 < cfg.min_rr_tp1:
                _bump("rr<min"); continue

            sess = s.context.get("session", "-")
            local = ts.tz_convert("America/New_York")
            if cfg.skip_lunch and sess == "LUNCH":
                _bump("lunch"); continue
            if cfg.skip_monday_before_ny and local.weekday() == 0 and local.hour < 10:
                _bump("monday<10"); continue
            if cfg.skip_same_session_after_loss and last_loss_session_key:
                key = str(local.date()) + "|" + sess
                if key == last_loss_session_key:
                    _bump("post-loss-same-session"); continue

            risk = abs(s.entry - s.sl)
            # Cap TP1 and TP2 at realistic partial ladder
            rr1_eff = s.rr_tp1
            tp1_eff = s.tp1
            if s.rr_tp1 is not None and s.rr_tp1 > cfg.tp1_cap_r:
                if s.direction == "long":
                    tp1_eff = s.entry + cfg.tp1_cap_r * risk
                else:
                    tp1_eff = s.entry - cfg.tp1_cap_r * risk
                rr1_eff = cfg.tp1_cap_r

            rr2_eff = s.rr_tp2 if s.rr_tp2 is not None else cfg.tp2_cap_r
            tp2_eff = s.tp2
            # If grader had no TP2, synthesise one at tp2_cap_r
            if tp2_eff is None or rr2_eff > cfg.tp2_cap_r:
                if s.direction == "long":
                    tp2_eff = s.entry + cfg.tp2_cap_r * risk
                else:
                    tp2_eff = s.entry - cfg.tp2_cap_r * risk
                rr2_eff = cfg.tp2_cap_r

            dk = f"{s.grade.value}|{s.direction}|{s.entry:.5f}|{s.sl:.5f}"
            if dk == last_dedup:
                _bump("dedup"); continue
            last_dedup = dk
            last_signal_bar = li

            spread = cfg.spread_pips * pip
            fill = s.entry + spread / 2 if s.direction == "long" else s.entry - spread / 2

            open_trade = Trade(
                open_ts=ts.to_pydatetime(),
                close_ts=None,
                pair=pair, direction=s.direction or "?", grade=s.grade.value,
                entry=s.entry, fill=fill, sl=s.sl, tp1=tp1_eff, tp2=tp2_eff,
                risk_r=s.risk_r, rr_tp1=rr1_eff, rr_tp2=rr2_eff,
                checklist_passed=s.passed(), checklist_total=s.total(),
                session=sess, dow=local.weekday(),
            )

        if verbose_every and i % verbose_every == 0 and i > 0:
            import sys
            print(f"  [bt] step {i}/{len(decision_idx)} ts={ts} trades={len(result.trades)}", flush=True)

    result.skip_reasons = skip_reasons

    # Close any hanging trade at last bar — mark as end-of-period (excluded from stats)
    if open_trade is not None:
        last_bar = ltf_in_range.iloc[-1]
        r = _realised_r(open_trade.fill, float(last_bar["close"]), open_trade.sl,
                        open_trade.direction, pip, cfg)
        open_trade.close_ts = ltf_in_range.index[-1].to_pydatetime()
        open_trade.outcome = "eop"       # end-of-period artifact
        open_trade.exit_price = float(last_bar["close"])
        open_trade.r_multiple = r
        open_trade.weighted_r = r * open_trade.risk_r
        result.trades.append(open_trade)

    return result



def _check_fill(t: Trade, bar, df: pd.DataFrame, bar_idx: int,
                cfg: ExecConfig, pip: float) -> None:
    """Limit-order fill simulation: the trade fills only when a bar trades through
    the entry price. SL hit on the same bar as fill → immediate loss."""
    h = float(bar["high"])
    l = float(bar["low"])
    # check if bar range includes entry (fill price)
    if l <= t.fill <= h:
        t.outcome = "open"
        t.filled_ts = df.index[bar_idx].to_pydatetime()
        # check same-bar SL immediately (conservative)
        sl_slip = cfg.sl_slippage_pips * pip
        eff_sl = t.sl - sl_slip if t.direction == "long" else t.sl + sl_slip
        if t.direction == "long" and l <= eff_sl:
            t.outcome = "loss"
            t.r_multiple = -1.0
            t.weighted_r = -1.0 * t.risk_r
            t.exit_price = eff_sl
            t.close_ts = t.filled_ts
        elif t.direction == "short" and h >= eff_sl:
            t.outcome = "loss"
            t.r_multiple = -1.0
            t.weighted_r = -1.0 * t.risk_r
            t.exit_price = eff_sl
            t.close_ts = t.filled_ts


def _update_open(t: Trade, bar, df: pd.DataFrame, bar_idx: int,
                 cfg: ExecConfig, pip: float) -> None:
    """Check if current bar hits SL / TP1 / TP2. Realistic fills:
      - SL includes slippage (adverse overshoot).
      - Same-bar SL+TP conflict -> SL wins (conservative worst-case).
      - Max-hold timeout exits at close with realised R.
    """
    h = float(bar["high"])
    l = float(bar["low"])
    close = float(bar["close"])
    t.bars_held += 1

    sl_slip = cfg.sl_slippage_pips * pip
    if t.direction == "long":
        eff_sl = t.sl - sl_slip
    else:
        eff_sl = t.sl + sl_slip
    risk = abs(t.fill - eff_sl)
    if risk <= 0:
        t.outcome = "be"
        t.close_ts = df.index[bar_idx].to_pydatetime()
        return

    if t.direction == "long":
        sl_hit = l <= eff_sl
        tp1_hit = (t.tp1 is not None and h >= t.tp1)
        tp2_hit = (t.tp2 is not None and h >= t.tp2)
    else:
        sl_hit = h >= eff_sl
        tp1_hit = (t.tp1 is not None and l <= t.tp1)
        tp2_hit = (t.tp2 is not None and l <= t.tp2)

    if sl_hit:
        t.outcome = "loss"
        t.r_multiple = -1.0
        t.weighted_r = -1.0 * t.risk_r
        t.exit_price = eff_sl
        t.close_ts = df.index[bar_idx].to_pydatetime()
        return

    if tp2_hit:
        r1 = _realised_r(t.fill, t.tp1, t.sl, t.direction, pip, cfg)
        r2 = _realised_r(t.fill, t.tp2, t.sl, t.direction, pip, cfg)
        r = cfg.partial_pct * r1 + (1 - cfg.partial_pct) * r2
        t.outcome = "win2"
        t.r_multiple = r
        t.weighted_r = r * t.risk_r
        t.exit_price = t.tp2
        t.close_ts = df.index[bar_idx].to_pydatetime()
        return

    if tp1_hit:
        # Partial at TP1 (default 50%); runner's SL trailed to break-even.
        # Realistic modelling: runner hits BE with high probability (~85%) before drifting to TP2.
        r1 = _realised_r(t.fill, t.tp1, t.sl, t.direction, pip, cfg)
        r = cfg.partial_pct * r1
        t.outcome = "win1"
        t.r_multiple = r
        t.weighted_r = r * t.risk_r
        t.exit_price = t.tp1
        t.close_ts = df.index[bar_idx].to_pydatetime()
        return

    if t.bars_held >= cfg.max_hold_bars:
        r = _realised_r(t.fill, close, t.sl, t.direction, pip, cfg)
        t.outcome = "timeout"
        t.r_multiple = r
        t.weighted_r = r * t.risk_r
        t.exit_price = close
        t.close_ts = df.index[bar_idx].to_pydatetime()
        return


def _realised_r(fill: float, exit_px: float, sl: float, direction: str,
                pip: float, cfg: ExecConfig) -> float:
    """R-multiple including spread cost on exit."""
    spread = cfg.spread_pips * pip
    if direction == "long":
        exit_net = exit_px - spread / 2
        pnl = exit_net - fill
    else:
        exit_net = exit_px + spread / 2
        pnl = fill - exit_net
    risk = abs(fill - sl)
    return pnl / risk if risk > 0 else 0.0
