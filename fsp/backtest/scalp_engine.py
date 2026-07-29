"""SCALP_MR backtester — M5 Bollinger-Band mean-reversion replay.

Purpose-built rather than reusing ``intraday_engine`` because SCALP_MR is a
single-target strategy (``tp2`` is None): the shared ``_update_open`` model
credits only ``partial_pct`` of TP1 and assumes a break-even runner, which
would systematically understate a single-target scalp's wins.

Cost model — every cost is adverse and expressed in pips:
  * entry fill = decision-bar close +/- spread/2
  * exit  fill = exit price       -/+ spread/2   (round trip pays full spread)
  * SL exit price overshoots the stop by ``sl_slippage_pips``
  * same-bar SL and TP -> SL wins (M5 bars hide the intra-bar sequence)

R denominator is the *intended* risk ``|signal.entry - signal.sl|``, because
that is what the live bridge sizes on. Costs therefore push a stopped-out
trade below -1R, which is the truth for a scalp whose stop is a few pips wide.

``entry_delay_bars`` models feed latency: the live loop runs on a yfinance
feed that lags ~15 min, so the signal computed on a bar is only actionable
about 3 M5 bars later, at whatever price the market has moved to by then.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

import pandas as pd

from fsp.backtest.engine import BacktestResult, Trade
from fsp.data.feed import default_feed
from fsp.signals.scalp_mr import scan_scalp_mr

log = logging.getLogger("fsp.scalp_bt")

# scan_scalp_mr needs >=25 bars and a 20-period band, so a 30-bar tail is
# mathematically identical to passing the whole history — and O(1) per bar.
SCAN_WINDOW = 30


@dataclass
class ScalpExecConfig:
    spread_pips: float = 0.3
    sl_slippage_pips: float = 0.3
    max_hold_bars: int = 3
    cooldown_bars: int = 2
    entry_delay_bars: int = 0


def _pip(pair: str) -> float:
    return 0.01 if "JPY" in pair else 0.0001


def _close_trade(t: Trade, outcome: str, exit_px: float,
                 bar_ts: pd.Timestamp, cfg: ScalpExecConfig, pip: float) -> None:
    half = cfg.spread_pips * pip / 2
    if t.direction == "long":
        pnl = (exit_px - half) - t.fill
    else:
        pnl = t.fill - (exit_px + half)
    risk = abs(t.entry - t.sl)
    r = pnl / risk if risk > 0 else 0.0
    t.outcome = outcome  # type: ignore[assignment]
    t.r_multiple = r
    t.weighted_r = r * t.risk_r
    t.exit_price = exit_px
    t.close_ts = bar_ts.to_pydatetime()


def _update(t: Trade, bar, bar_ts: pd.Timestamp,
            cfg: ScalpExecConfig, pip: float) -> None:
    high, low, close = float(bar["high"]), float(bar["low"]), float(bar["close"])
    t.bars_held += 1
    slip = cfg.sl_slippage_pips * pip

    if t.direction == "long":
        eff_sl = t.sl - slip
        sl_hit = low <= eff_sl
        tp_hit = t.tp1 is not None and high >= t.tp1
    else:
        eff_sl = t.sl + slip
        sl_hit = high >= eff_sl
        tp_hit = t.tp1 is not None and low <= t.tp1

    if sl_hit:
        _close_trade(t, "loss", eff_sl, bar_ts, cfg, pip)
    elif tp_hit:
        _close_trade(t, "win1", float(t.tp1), bar_ts, cfg, pip)
    elif t.bars_held >= cfg.max_hold_bars:
        _close_trade(t, "timeout", close, bar_ts, cfg, pip)


def _enter(sig, ts: pd.Timestamp, bar_close: float,
           cfg: ScalpExecConfig, pip: float) -> Trade:
    """Market entry at the close of the fill bar, paying half the spread."""
    half = cfg.spread_pips * pip / 2
    fill = bar_close + half if sig.direction == "long" else bar_close - half
    t = Trade(
        open_ts=ts.to_pydatetime(),
        close_ts=None,
        pair=sig.pair,
        direction=sig.direction,
        grade=sig.strategy,
        entry=sig.entry,
        fill=fill,
        sl=sig.sl,
        tp1=sig.tp1,
        tp2=None,
        risk_r=sig.risk_r,
        rr_tp1=sig.rr_tp1,
        rr_tp2=None,
        checklist_passed=0,
        checklist_total=0,
        session=str(sig.context.get("session", "-")),
        dow=ts.weekday(),
    )
    t.outcome = "open"
    t.filled_ts = ts.to_pydatetime()
    return t


def run_scalp_backtest(
    pair: str,
    start: datetime,
    end: datetime,
    feed_kind: str = "duka",
    cfg: ScalpExecConfig | None = None,
    m5: pd.DataFrame | None = None,
    verbose: bool = False,
) -> BacktestResult:
    """Replay SCALP_MR bar-by-bar over [start, end] for one pair.

    ``m5`` lets a caller pass pre-loaded bars so a spread sweep reuses one
    download instead of re-fetching per configuration.
    """
    cfg = cfg or ScalpExecConfig()
    pip = _pip(pair)

    if m5 is None:
        f = default_feed(feed_kind)
        m5 = f.history(pair, "M5", start - timedelta(days=4), end)

    result = BacktestResult(start=start, end=end)
    if m5.empty:
        return result

    _s, _e = pd.Timestamp(start), pd.Timestamp(end)
    idx = m5.index
    open_trade: Trade | None = None
    pending: tuple[int, object] | None = None
    last_signal_i = -(10**9)
    last_entry_key: str | None = None
    n_fired = 0

    for i in range(len(m5)):
        ts = idx[i]
        if ts > _e:
            break
        bar = m5.iloc[i]

        if open_trade is not None:
            _update(open_trade, bar, ts, cfg, pip)
            if open_trade.outcome != "open":
                result.trades.append(open_trade)
                open_trade = None

        # A queued signal fills once its latency has elapsed. If a trade is
        # still running at that point the signal is dropped, matching the live
        # bridge (one position per pair per strategy).
        if pending is not None and i >= pending[0]:
            sig = pending[1]
            pending = None
            if open_trade is None:
                open_trade = _enter(sig, ts, float(bar["close"]), cfg, pip)
                continue

        if open_trade is not None or pending is not None:
            continue
        if ts < _s or i < SCAN_WINDOW:
            continue
        if i - last_signal_i < cfg.cooldown_bars:
            continue

        try:
            sig = scan_scalp_mr(pair, m5.iloc[i - SCAN_WINDOW + 1: i + 1])
        except Exception as exc:  # noqa: BLE001
            log.debug("scalp scan err at %s: %s", ts, exc)
            continue
        if sig is None:
            continue

        entry_key = f"{sig.direction}|{sig.entry:.5f}"
        if entry_key == last_entry_key:
            continue
        last_entry_key = entry_key
        last_signal_i = i
        n_fired += 1

        if cfg.entry_delay_bars <= 0:
            open_trade = _enter(sig, ts, float(bar["close"]), cfg, pip)
        else:
            pending = (i + cfg.entry_delay_bars, sig)

    if open_trade is not None and open_trade.outcome == "open":
        _close_trade(open_trade, "eop", float(m5.iloc[-1]["close"]),
                     idx[-1], cfg, pip)
        result.trades.append(open_trade)

    if verbose:
        st = result.stats()
        print(f"  {pair}: fired={n_fired} closed={st.get('total', 0)}", flush=True)

    return result


def pip_pnl(t: Trade, pip: float) -> float:
    """Net pips for a closed trade, derived from its R and intended risk."""
    return t.r_multiple * abs(t.entry - t.sl) / pip
