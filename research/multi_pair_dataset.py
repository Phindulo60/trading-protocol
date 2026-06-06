"""
Multi-pair training dataset generator for the GBM meta-model.

Walks the production scanners (scan_trend_rsi + scan_asia_hl) bar-by-bar
across 7 majors, captures features at each signal fire, simulates the
trade outcome, and writes (features, outcome) rows to parquet.

Reuses production scanner code → backtest matches live by construction.

Output: research/training_data.parquet
"""
from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

# Make the project root importable
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))

import numpy as np
import pandas as pd

from fsp.data.feed import DukascopyFeed
from fsp.ml.features import FeatureExtractor, FEATURE_NAMES
from fsp.signals.alpha import scan_trend_rsi
from fsp.signals.asia_hl import scan_asia_hl

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("multi_pair_dataset")
# Reduce noise from feature extractor
logging.getLogger("fsp.ml.features").setLevel(logging.WARNING)
logging.getLogger("fsp.data.feed").setLevel(logging.WARNING)


# ── Config ────────────────────────────────────────────────────────────────────

PAIRS = ["USDCAD", "EURUSD", "GBPUSD", "AUDUSD", "USDJPY", "EURJPY", "GBPJPY"]
START_DATE = datetime(2025, 6, 1, tzinfo=timezone.utc)  # 12 months back
END_DATE = datetime(2026, 6, 1, tzinfo=timezone.utc)

OUT_PATH = HERE / "training_data.parquet"
CACHE_DIR = HERE / "_cache"
CACHE_DIR.mkdir(exist_ok=True)


# ── Trade simulator (matches production: fill at signal close, walk-forward) ──

def simulate_trade(
    sig, future_bars: pd.DataFrame, pip: float,
    spread_pips: float = 0.3, sl_slip_pips: float = 0.3,
) -> dict:
    """Return outcome dict: outcome, r_realized, exit_ts, bars_held."""
    direction = sig.direction
    entry = sig.entry
    fill = entry + (spread_pips * pip / 2 if direction == "long"
                    else -spread_pips * pip / 2)
    sl = sig.sl
    tp1 = sig.tp1
    tp2 = sig.tp2 if sig.tp2 is not None else (sig.tp1 + abs(sig.tp1 - sig.entry) * 0.5)
    risk = abs(fill - sl)
    if risk == 0:
        return {"outcome": "invalid", "r_realized": 0.0, "exit_ts": None, "bars_held": 0}

    max_hold = sig.context.get("max_hold_bars", 16)
    hit_tp1 = False
    runner_sl = sl

    for i in range(min(max_hold, len(future_bars))):
        bar = future_bars.iloc[i]
        hi, lo = float(bar["high"]), float(bar["low"])
        bar_ts = future_bars.index[i]

        if direction == "long":
            if not hit_tp1:
                if lo <= sl:  # SL first if both hit (conservative)
                    return {"outcome": "loss", "r_realized": -1.0,
                            "exit_ts": bar_ts.isoformat(), "bars_held": i + 1}
                if hi >= tp1:
                    hit_tp1 = True
                    runner_sl = fill  # to BE
                    if hi >= tp2:
                        r = 0.5 * (tp1 - fill) / risk + 0.5 * (tp2 - fill) / risk
                        return {"outcome": "win", "r_realized": float(r),
                                "exit_ts": bar_ts.isoformat(), "bars_held": i + 1}
            else:
                if lo <= runner_sl:
                    r1 = 0.5 * (tp1 - fill) / risk
                    r2 = 0.5 * (runner_sl - fill) / risk
                    outcome = "win" if (r1 + r2) > 0 else "be"
                    return {"outcome": outcome, "r_realized": float(r1 + r2),
                            "exit_ts": bar_ts.isoformat(), "bars_held": i + 1}
                if hi >= tp2:
                    r = 0.5 * (tp1 - fill) / risk + 0.5 * (tp2 - fill) / risk
                    return {"outcome": "win", "r_realized": float(r),
                            "exit_ts": bar_ts.isoformat(), "bars_held": i + 1}
        else:  # short
            if not hit_tp1:
                if hi >= sl:
                    return {"outcome": "loss", "r_realized": -1.0,
                            "exit_ts": bar_ts.isoformat(), "bars_held": i + 1}
                if lo <= tp1:
                    hit_tp1 = True
                    runner_sl = fill
                    if lo <= tp2:
                        r = 0.5 * (fill - tp1) / risk + 0.5 * (fill - tp2) / risk
                        return {"outcome": "win", "r_realized": float(r),
                                "exit_ts": bar_ts.isoformat(), "bars_held": i + 1}
            else:
                if hi >= runner_sl:
                    r1 = 0.5 * (fill - tp1) / risk
                    r2 = 0.5 * (fill - runner_sl) / risk
                    outcome = "win" if (r1 + r2) > 0 else "be"
                    return {"outcome": outcome, "r_realized": float(r1 + r2),
                            "exit_ts": bar_ts.isoformat(), "bars_held": i + 1}
                if lo <= tp2:
                    r = 0.5 * (fill - tp1) / risk + 0.5 * (fill - tp2) / risk
                    return {"outcome": "win", "r_realized": float(r),
                            "exit_ts": bar_ts.isoformat(), "bars_held": i + 1}

    # Timeout — exit at last bar close
    if len(future_bars) > 0:
        last_idx = min(max_hold - 1, len(future_bars) - 1)
        last_close = float(future_bars["close"].iloc[last_idx])
        if not hit_tp1:
            r = ((last_close - fill) / risk if direction == "long"
                 else (fill - last_close) / risk)
        else:
            r1 = (0.5 * (tp1 - fill) / risk if direction == "long"
                  else 0.5 * (fill - tp1) / risk)
            r2 = (0.5 * (last_close - fill) / risk if direction == "long"
                  else 0.5 * (fill - last_close) / risk)
            r = r1 + r2
        return {"outcome": "timeout", "r_realized": float(r),
                "exit_ts": future_bars.index[last_idx].isoformat(),
                "bars_held": last_idx + 1}
    return {"outcome": "no_data", "r_realized": 0.0, "exit_ts": None, "bars_held": 0}


# ── Data loader ──────────────────────────────────────────────────────────────

def load_pair_data(pair: str, start: datetime, end: datetime
                   ) -> dict[str, pd.DataFrame]:
    """Load M15/H1/H4/D for one pair. Uses Dukascopy cache."""
    feed = DukascopyFeed()
    cache_key = f"{pair}_{start:%Y%m%d}_{end:%Y%m%d}"

    out = {}
    for tf, days_buffer in [("M15", 60), ("H1", 60), ("D", 60)]:
        cache_file = CACHE_DIR / f"{cache_key}_{tf}.parquet"
        if cache_file.exists():
            out[tf] = pd.read_parquet(cache_file)
            continue
        # Add buffer for indicator warm-up
        s = start - timedelta(days=days_buffer)
        log.info("  fetching %s %s [%s → %s]...", pair, tf,
                 s.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
        df = feed.history(pair, tf, s, end)  # type: ignore[arg-type]
        if df.empty:
            log.warning("  no data for %s %s", pair, tf)
            out[tf] = pd.DataFrame()
            continue
        df.to_parquet(cache_file)
        out[tf] = df

    # Build H4 by resampling M15 (matches scanner.py)
    if not out["M15"].empty:
        out["H4"] = out["M15"].resample("4h").agg({
            "open": "first", "high": "max", "low": "min",
            "close": "last", "volume": "sum",
        }).dropna()
    else:
        out["H4"] = pd.DataFrame()
    return out


# ── Simulated journal for backtest feature extraction ─────────────────────────

class SimulatedJournal:
    """Holds trades simulated so far so feature extractor sees correct state."""

    def __init__(self):
        self.trades: list[dict] = []

    def add(self, sig, outcome: dict) -> None:
        self.trades.append({
            "ts": sig.ts,
            "pair": sig.pair,
            "strategy": sig.strategy,
            "direction": sig.direction,
            "outcome": outcome["outcome"],
            "r_multiple": outcome["r_realized"],
        })

    def query(self, strategy: str, before: datetime, limit: int) -> list[dict]:
        before_iso = before.isoformat() if hasattr(before, "isoformat") else str(before)
        rows = [t for t in self.trades
                if t["strategy"] == strategy and t["ts"] < before_iso]
        rows.sort(key=lambda r: r["ts"], reverse=True)
        return rows[:limit]


# ── Backtest loop per pair ────────────────────────────────────────────────────

def backtest_pair(
    pair: str,
    start: datetime,
    end: datetime,
    cooldown_bars: int = 4,
) -> list[dict]:
    """Walk M15 bars, fire scanners, capture features+outcomes. Returns rows."""
    log.info("Loading data for %s...", pair)
    data = load_pair_data(pair, start, end)
    m15_full, h1_full, h4_full = data["M15"], data["H1"], data["H4"]

    if m15_full.empty:
        log.warning("Skipping %s — no M15 data", pair)
        return []

    log.info("  M15 bars=%d, H1=%d, H4=%d", len(m15_full), len(h1_full), len(h4_full))
    log.info("  Range: %s → %s", m15_full.index[0], m15_full.index[-1])

    # Filter to actual backtest window (after warm-up)
    valid_mask = m15_full.index >= start
    valid_indices = np.where(valid_mask)[0]
    if len(valid_indices) == 0:
        log.warning("No bars in valid range")
        return []
    first_idx = max(int(valid_indices[0]), 60)  # need 60 bars for indicators

    pip = 0.01 if "JPY" in pair else 0.0001
    extractor = FeatureExtractor()
    journal = SimulatedJournal()

    rows: list[dict] = []
    last_signal_idx = {-1: -999}  # per-strategy cooldown
    open_until_idx = -1  # block while in trade
    n_total = len(m15_full)
    last_log_t = time.time()

    for i in range(first_idx, n_total):
        if i < open_until_idx:
            continue

        ts = m15_full.index[i]
        m15_slice = m15_full.iloc[:i + 1]
        h1_slice = h1_full[h1_full.index <= ts] if not h1_full.empty else pd.DataFrame()
        h4_slice = h4_full[h4_full.index <= ts] if not h4_full.empty else pd.DataFrame()

        # Run both scanners
        for scanner_name, scanner_fn, args in [
            ("TREND_RSI", scan_trend_rsi, (m15_slice, h4_slice)),
            ("ASIA_HL", scan_asia_hl, (m15_slice, h4_slice)),
        ]:
            cd_key = scanner_name
            if i - last_signal_idx.get(cd_key, -999) < cooldown_bars:
                continue
            try:
                sig = scanner_fn(pair, *args)
            except Exception as e:
                log.debug("Scanner %s failed at %s: %s", scanner_name, ts, e)
                continue
            if sig is None:
                continue

            # Simulate trade outcome on future bars
            future = m15_full.iloc[i + 1:]
            outcome = simulate_trade(sig, future, pip)
            if outcome["outcome"] in ("invalid", "no_data"):
                continue

            # Extract features at signal time, using SIMULATED journal so far
            try:
                fs = extractor.extract(
                    pair=pair,
                    strategy=sig.strategy,
                    direction=sig.direction,
                    ts=ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts,
                    m15=m15_slice,
                    h1=h1_slice if not h1_slice.empty else None,
                    h4=h4_slice if not h4_slice.empty else None,
                    llm_result=None,  # backtest has no LLM
                    journal_query=journal.query,
                )
                features = fs.to_dict()
            except Exception as e:
                log.warning("Feature extraction failed at %s: %s", ts, e)
                continue

            # Build training row.
            # Label: "profitable" = r > 0 (includes partial wins via timeout).
            # Pure TP-hit "won_tp" kept as alt label for stricter filter training.
            row = {
                **features,
                "ts": ts.isoformat(),
                "outcome": outcome["outcome"],
                "r_realized": outcome["r_realized"],
                "won": 1 if outcome["r_realized"] > 0 else 0,
                "won_tp": 1 if outcome["outcome"] == "win" else 0,
                "exit_ts": outcome["exit_ts"],
                "bars_held": outcome["bars_held"],
                "entry": sig.entry,
                "sl": sig.sl,
                "tp1": sig.tp1,
                "rr_tp1": sig.rr_tp1,
            }
            rows.append(row)
            journal.add(sig, outcome)
            last_signal_idx[cd_key] = i
            open_until_idx = max(open_until_idx, i + outcome["bars_held"] + 1)

        # Progress log every 10s
        if time.time() - last_log_t > 10:
            pct = (i - first_idx) / max(1, n_total - first_idx) * 100
            log.info("  %s: bar %d/%d (%.0f%%) — %d trades captured",
                     pair, i, n_total, pct, len(rows))
            last_log_t = time.time()

    log.info("  %s: COMPLETE — %d trades", pair, len(rows))
    return rows


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 70)
    log.info("Multi-pair training dataset generation")
    log.info("Pairs: %s", PAIRS)
    log.info("Range: %s → %s", START_DATE, END_DATE)
    log.info("=" * 70)

    t0 = time.time()
    all_rows: list[dict] = []

    for pair in PAIRS:
        try:
            rows = backtest_pair(pair, START_DATE, END_DATE)
            all_rows.extend(rows)
        except Exception as e:
            log.exception("Pair %s failed: %s", pair, e)

    if not all_rows:
        log.error("No training rows generated — aborting")
        return

    df = pd.DataFrame(all_rows)
    df.to_parquet(OUT_PATH)

    # Summary
    log.info("")
    log.info("=" * 70)
    log.info("DATASET SUMMARY")
    log.info("=" * 70)
    log.info("Total trades: %d", len(df))
    log.info("Output: %s", OUT_PATH)
    log.info("")

    # Per-pair / per-strategy breakdown
    summary = (df.groupby(["pair", "strategy"])
                 .agg(n=("won", "size"),
                      wr=("won", "mean"),
                      total_r=("r_realized", "sum"),
                      avg_r=("r_realized", "mean")))
    log.info("\n%s", summary.round(3).to_string())

    log.info("")
    log.info("Win-rate distribution: %.1f%% wins, %.1f%% losses, %.1f%% timeouts",
             (df["outcome"] == "win").mean() * 100,
             (df["outcome"] == "loss").mean() * 100,
             (df["outcome"] == "timeout").mean() * 100)
    log.info("Total R across all trades: %+.1f", df["r_realized"].sum())
    log.info("Elapsed: %.1f min", (time.time() - t0) / 60)


if __name__ == "__main__":
    main()
