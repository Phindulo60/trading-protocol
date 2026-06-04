"""Feature extractor for the GBM meta-model.

Computes 28 features at signal-fire time. Features are intentionally minimal
and explainable — small dataset means we cannot afford 100+ features.

CRITICAL: NO future data may be used. Features must be computable using ONLY
information available at the signal bar's close.

Feature schema:
    Categorical (5):  strategy, direction, pair, session, dow
    Technical (10):   rsi_at_entry, rsi_depth, atr_pctl_60d, adx_h4,
                      h4_trend_score, h1_trend_score, m15_atr_norm,
                      adr_pct_used, bars_since_extreme, range_compress
    Strategy (5):     strat_wr_last_20, strat_streak_signed,
                      strat_net_r_last_20, pair_strat_wr_last_10,
                      days_since_last_signal
    Calendar (4):     mins_to_next_high_event, high_events_in_24h,
                      dxy_change_24h_pct, dxy_trend_aligned
    LLM (4):          llm_confidence, llm_decision_take,
                      llm_decision_skip, llm_decision_reduce
"""
from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger("fsp.ml.features")

# Feature schema kept as a constant — used by trainer/predictor for ordering
FEATURE_NAMES: list[str] = [
    # Categorical
    "strategy", "direction", "pair", "session", "dow",
    # Technical
    "rsi_at_entry", "rsi_depth", "atr_pctl_60d", "adx_h4",
    "h4_trend_score", "h1_trend_score", "m15_atr_norm",
    "adr_pct_used", "bars_since_extreme", "range_compress",
    # Strategy state
    "strat_wr_last_20", "strat_streak_signed",
    "strat_net_r_last_20", "pair_strat_wr_last_10",
    "days_since_last_signal",
    # Calendar / cross-asset
    "mins_to_next_high_event", "high_events_in_24h",
    "dxy_change_24h_pct", "dxy_trend_aligned",
    # LLM
    "llm_confidence", "llm_decision_take",
    "llm_decision_skip", "llm_decision_reduce",
]

CATEGORICAL_FEATURES = ["strategy", "direction", "pair", "session", "dow"]


@dataclass
class FeatureSet:
    """Strongly-typed feature row. Convert to dict for storage."""
    # Categorical
    strategy: str
    direction: str  # "long" | "short"
    pair: str
    session: str
    dow: str

    # Technical
    rsi_at_entry: float
    rsi_depth: float
    atr_pctl_60d: float
    adx_h4: float
    h4_trend_score: float
    h1_trend_score: float
    m15_atr_norm: float
    adr_pct_used: float
    bars_since_extreme: int
    range_compress: float

    # Strategy state
    strat_wr_last_20: float
    strat_streak_signed: int
    strat_net_r_last_20: float
    pair_strat_wr_last_10: float
    days_since_last_signal: float

    # Calendar / cross-asset
    mins_to_next_high_event: float
    high_events_in_24h: int
    dxy_change_24h_pct: float
    dxy_trend_aligned: int

    # LLM
    llm_confidence: float
    llm_decision_take: int
    llm_decision_skip: int
    llm_decision_reduce: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ── Helper computations ──────────────────────────────────────────────────────

def _rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1 / length, adjust=False).mean()
    roll_down = down.ewm(alpha=1 / length, adjust=False).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / length, adjust=False).mean()


def _adx(df: pd.DataFrame, length: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / length, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / length, adjust=False).mean() / atr.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=1 / length, adjust=False).mean() / atr.replace(0, np.nan)
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) * 100
    return dx.ewm(alpha=1 / length, adjust=False).mean()


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _session_at(ts: datetime) -> str:
    h = ts.hour
    if h < 8:   return "ASIA"
    if h < 12:  return "LO"
    if h < 16:  return "NY-AM"
    if h < 21:  return "NY-PM"
    return "OFF"


def _safe_float(x: Any, default: float = 0.0) -> float:
    """Convert anything to float, returning default on NaN/None/error."""
    if x is None:
        return default
    try:
        f = float(x)
        return default if (math.isnan(f) or math.isinf(f)) else f
    except (TypeError, ValueError):
        return default


# ── Sub-feature builders ─────────────────────────────────────────────────────

def _technical_features(
    pair: str, direction: str,
    m15: pd.DataFrame, h1: pd.DataFrame | None, h4: pd.DataFrame | None,
) -> dict[str, Any]:
    """Compute the 10 technical features. Tolerant to missing data."""
    out = {
        "rsi_at_entry": 50.0, "rsi_depth": 0.0,
        "atr_pctl_60d": 0.5, "adx_h4": 20.0,
        "h4_trend_score": 0.0, "h1_trend_score": 0.0,
        "m15_atr_norm": 1.0, "adr_pct_used": 0.0,
        "bars_since_extreme": 0, "range_compress": 1.0,
    }
    if m15 is None or len(m15) < 30:
        return out

    # RSI on M15 (entry RSI is what TREND_RSI fires on)
    rsi = _rsi(m15["close"], 14)
    rsi_now = _safe_float(rsi.iloc[-1], 50.0)
    out["rsi_at_entry"] = rsi_now
    threshold = 38 if direction == "long" else 62
    out["rsi_depth"] = abs(rsi_now - threshold)

    # ATR percentile vs 60-day window (~5760 M15 bars)
    atr_m15 = _atr(m15, 14)
    if len(atr_m15.dropna()) >= 60:
        recent = atr_m15.dropna().tail(min(5760, len(atr_m15.dropna())))
        atr_now = _safe_float(atr_m15.iloc[-1])
        if atr_now > 0:
            out["atr_pctl_60d"] = float((recent < atr_now).mean())

    # M15 range compression: current bar range vs 20-bar avg range
    bar_range = m15["high"].iloc[-1] - m15["low"].iloc[-1]
    avg_range = (m15["high"] - m15["low"]).tail(20).mean()
    if avg_range > 0:
        out["range_compress"] = _safe_float(bar_range / avg_range, 1.0)

    # Bars since 20-bar extreme (high for long, low for short)
    if direction == "long":
        recent_max = m15["high"].tail(20)
        out["bars_since_extreme"] = int(20 - 1 - recent_max.values.argmax())
    else:
        recent_min = m15["low"].tail(20)
        out["bars_since_extreme"] = int(20 - 1 - recent_min.values.argmin())

    # H4 trend score: (close - ema10) / atr  (signed)
    if h4 is not None and len(h4) >= 20:
        ema10 = _ema(h4["close"], 10)
        atr_h4 = _atr(h4, 14)
        diff = h4["close"].iloc[-1] - ema10.iloc[-1]
        atr_v = _safe_float(atr_h4.iloc[-1])
        if atr_v > 0:
            out["h4_trend_score"] = _safe_float(diff / atr_v)
        # ADX
        adx = _adx(h4, 14)
        out["adx_h4"] = _safe_float(adx.iloc[-1], 20.0)

    # H1 trend score
    if h1 is not None and len(h1) >= 20:
        ema20 = _ema(h1["close"], 20)
        atr_h1 = _atr(h1, 14)
        diff = h1["close"].iloc[-1] - ema20.iloc[-1]
        atr_v = _safe_float(atr_h1.iloc[-1])
        if atr_v > 0:
            out["h1_trend_score"] = _safe_float(diff / atr_v)

    # M15 ATR normalised by H4 ATR (relative volatility)
    if h4 is not None and len(h4) >= 20:
        atr_h4_v = _safe_float(_atr(h4, 14).iloc[-1])
        atr_m15_v = _safe_float(atr_m15.iloc[-1])
        if atr_h4_v > 0:
            # Convert: 1 H4 = 16 M15 bars, so equivalent M15 ATR in H4 terms
            out["m15_atr_norm"] = _safe_float((atr_m15_v * 16) / atr_h4_v, 1.0)

    # ADR usage (today's range / average daily range)
    if h1 is not None and len(h1) >= 24:
        # Last 24 H1 bars = today
        today = h1.tail(24)
        today_range = today["high"].max() - today["low"].min()
        # ADR(5) — last 5 calendar days
        if len(h1) >= 24 * 5:
            daily_ranges = []
            for i in range(5):
                day = h1.iloc[-(i + 1) * 24:-i * 24] if i > 0 else h1.tail(24)
                if len(day) > 0:
                    daily_ranges.append(day["high"].max() - day["low"].min())
            if daily_ranges:
                adr5 = float(np.mean(daily_ranges))
                if adr5 > 0:
                    out["adr_pct_used"] = _safe_float(today_range / adr5)

    return out


def _strategy_state_features(
    pair: str, strategy: str, ts: datetime, journal_query
) -> dict[str, Any]:
    """Pull strategy state from the journal at signal time. journal_query is a callable."""
    out = {
        "strat_wr_last_20": 0.5,
        "strat_streak_signed": 0,
        "strat_net_r_last_20": 0.0,
        "pair_strat_wr_last_10": 0.5,
        "days_since_last_signal": 7.0,
    }
    try:
        rows = journal_query(strategy=strategy, before=ts, limit=20)
    except Exception as e:
        log.warning("Strategy-state query failed: %s", e)
        return out

    # Filter resolved (non-None outcome)
    resolved = [r for r in rows if r.get("outcome") in ("win", "loss")]
    if resolved:
        wins = sum(1 for r in resolved if r["outcome"] == "win")
        out["strat_wr_last_20"] = wins / len(resolved)
        out["strat_net_r_last_20"] = sum(_safe_float(r.get("r_multiple")) for r in resolved)
        # Streak: signed count of consecutive same outcomes (most recent first)
        last_outcome = resolved[0]["outcome"]
        streak = 1
        for r in resolved[1:]:
            if r["outcome"] == last_outcome:
                streak += 1
            else:
                break
        out["strat_streak_signed"] = streak if last_outcome == "win" else -streak

    # Pair-specific WR (last 10 trades of this strategy on this pair)
    pair_rows = [r for r in resolved if r.get("pair") == pair][:10]
    if pair_rows:
        pair_wins = sum(1 for r in pair_rows if r["outcome"] == "win")
        out["pair_strat_wr_last_10"] = pair_wins / len(pair_rows)

    # Days since last signal of this strategy
    if rows:
        last_ts_str = rows[0].get("ts", "")
        try:
            last_ts = pd.Timestamp(last_ts_str)
            if last_ts.tzinfo is None:
                last_ts = last_ts.tz_localize("UTC")
            delta_days = (pd.Timestamp(ts).tz_convert("UTC")
                          - last_ts.tz_convert("UTC")).total_seconds() / 86400
            out["days_since_last_signal"] = float(max(0.0, delta_days))
        except Exception:
            pass

    return out


def _calendar_features(pair: str, ts: datetime) -> dict[str, Any]:
    """Calendar + DXY features. Best-effort — returns defaults on failure."""
    out = {
        "mins_to_next_high_event": 240.0,
        "high_events_in_24h": 0,
        "dxy_change_24h_pct": 0.0,
        "dxy_trend_aligned": 0,
    }
    try:
        from fsp.llm.context import upcoming_events
        events = upcoming_events(pair, hours_ahead=24.0)
        high = [e for e in events if e.get("impact", "").lower() == "high"]
        if high:
            mins = [e["minutes_away"] for e in high if e["minutes_away"] >= 0]
            out["mins_to_next_high_event"] = float(min(mins)) if mins else 240.0
        out["high_events_in_24h"] = len([
            e for e in events if e.get("impact", "").lower() in ("high", "medium")
        ])
    except Exception as e:
        log.debug("Calendar features unavailable: %s", e)

    # DXY change/alignment — optional, skip if data unavailable
    try:
        from fsp.data.feed import default_feed
        f = default_feed("yf")
        end = ts
        df = f.history("DXY", "H1", end - timedelta(days=2), end)
        if df is not None and len(df) >= 24:
            now = float(df["close"].iloc[-1])
            prior = float(df["close"].iloc[-24])
            if prior > 0:
                out["dxy_change_24h_pct"] = (now - prior) / prior * 100
            # DXY rising = USD bullish. For USDxxx pairs, long aligns with rising DXY
            # For xxxUSD pairs, short aligns with rising DXY
            usd_first = pair.startswith("USD")
            dxy_up = out["dxy_change_24h_pct"] > 0
            # We don't know direction here — caller injects later via .dxy_trend_aligned
    except Exception as e:
        log.debug("DXY features unavailable: %s", e)
    return out


def _llm_features(llm_result) -> dict[str, Any]:
    """Convert LLM ValidationResult into 4 features."""
    out = {
        "llm_confidence": 0.5,
        "llm_decision_take": 0,
        "llm_decision_skip": 0,
        "llm_decision_reduce": 0,
    }
    if llm_result is None:
        return out
    out["llm_confidence"] = _safe_float(getattr(llm_result, "confidence", 0.5), 0.5)
    decision = getattr(llm_result, "decision", "TAKE")
    if decision == "TAKE":
        out["llm_decision_take"] = 1
    elif decision == "SKIP":
        out["llm_decision_skip"] = 1
    elif decision == "REDUCE":
        out["llm_decision_reduce"] = 1
    return out


# ── Main extractor ───────────────────────────────────────────────────────────

class FeatureExtractor:
    """Extract 28 features for a signal at signal-fire time.

    Designed to never raise — all sub-extractors fall back to safe defaults
    so a feature-extraction failure does not break the live signal pipeline.
    """

    def extract(
        self,
        pair: str,
        strategy: str,
        direction: str,
        ts: datetime,
        m15: pd.DataFrame,
        h1: pd.DataFrame | None = None,
        h4: pd.DataFrame | None = None,
        llm_result=None,
        journal_query=None,
    ) -> FeatureSet:
        """Build a FeatureSet. Tolerant to missing inputs.

        Args:
            pair: e.g. "USDCAD"
            strategy: "TREND_RSI" | "ASIA_HL"
            direction: "long" | "short"
            ts: signal bar timestamp (UTC)
            m15: M15 OHLC up to and including signal bar
            h1: H1 OHLC (optional)
            h4: H4 OHLC (optional)
            llm_result: ValidationResult or None
            journal_query: callable(strategy, before, limit) -> list[dict]; if None,
                           falls back to direct journal lookup
        """
        # Categorical
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        dow_names = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri",
                     5: "Sat", 6: "Sun"}
        cat = {
            "strategy": strategy,
            "direction": direction,
            "pair": pair,
            "session": _session_at(ts),
            "dow": dow_names[ts.weekday()],
        }

        # Technical
        tech = _technical_features(pair, direction, m15, h1, h4)

        # Strategy state
        if journal_query is None:
            journal_query = _default_journal_query
        strat = _strategy_state_features(pair, strategy, ts, journal_query)

        # Calendar
        cal = _calendar_features(pair, ts)
        # Compute dxy_trend_aligned from calendar pct + direction + pair
        try:
            usd_first = pair.startswith("USD")
            dxy_up = cal["dxy_change_24h_pct"] > 0.1  # 0.1% threshold
            dxy_dn = cal["dxy_change_24h_pct"] < -0.1
            if usd_first and direction == "long" and dxy_up:
                cal["dxy_trend_aligned"] = 1
            elif usd_first and direction == "short" and dxy_dn:
                cal["dxy_trend_aligned"] = 1
            elif not usd_first and direction == "long" and dxy_dn:
                cal["dxy_trend_aligned"] = 1
            elif not usd_first and direction == "short" and dxy_up:
                cal["dxy_trend_aligned"] = 1
        except Exception:
            pass

        # LLM
        llm = _llm_features(llm_result)

        return FeatureSet(**cat, **tech, **strat, **cal, **llm)


def _default_journal_query(strategy: str, before: datetime, limit: int) -> list[dict]:
    """Default implementation: pull from intraday_signals journal."""
    try:
        from fsp.journal.db import conn, migrate
        with conn() as c:
            migrate(c)
            rows = c.execute(
                "SELECT pair, strategy, direction, outcome, r_multiple, ts "
                "FROM intraday_signals "
                "WHERE strategy=? AND ts<? "
                "ORDER BY ts DESC LIMIT ?",
                (strategy, before.isoformat(), limit),
            ).fetchall()
        return [
            {"pair": r[0], "strategy": r[1], "direction": r[2],
             "outcome": r[3], "r_multiple": r[4], "ts": r[5]}
            for r in rows
        ]
    except Exception as e:
        log.warning("Journal query failed: %s", e)
        return []
