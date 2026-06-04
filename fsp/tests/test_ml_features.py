"""Tests for ML feature extraction."""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from fsp.ml.features import (
    CATEGORICAL_FEATURES,
    FEATURE_NAMES,
    FeatureExtractor,
    FeatureSet,
    _adx,
    _atr,
    _ema,
    _llm_features,
    _rsi,
    _safe_float,
    _session_at,
    _strategy_state_features,
    _technical_features,
)


# ── Schema ──────────────────────────────────────────────────────────────────

def test_feature_names_count():
    """Spec promises 28 features."""
    assert len(FEATURE_NAMES) == 28


def test_feature_names_unique():
    assert len(set(FEATURE_NAMES)) == len(FEATURE_NAMES)


def test_categorical_subset():
    for c in CATEGORICAL_FEATURES:
        assert c in FEATURE_NAMES


# ── Helpers ─────────────────────────────────────────────────────────────────

def test_safe_float_handles_none():
    assert _safe_float(None, 1.5) == 1.5


def test_safe_float_handles_nan():
    assert _safe_float(float("nan"), 1.5) == 1.5


def test_safe_float_handles_inf():
    assert _safe_float(float("inf"), 1.5) == 1.5


def test_safe_float_passthrough():
    assert _safe_float(3.14) == 3.14


def test_session_at_asia():
    assert _session_at(datetime(2026, 6, 5, 4, 0, tzinfo=timezone.utc)) == "ASIA"


def test_session_at_lo():
    assert _session_at(datetime(2026, 6, 5, 10, 0, tzinfo=timezone.utc)) == "LO"


def test_session_at_ny_am():
    assert _session_at(datetime(2026, 6, 5, 14, 0, tzinfo=timezone.utc)) == "NY-AM"


def test_session_at_off():
    assert _session_at(datetime(2026, 6, 5, 22, 0, tzinfo=timezone.utc)) == "OFF"


# ── Indicator math (sanity checks) ──────────────────────────────────────────

def _toy_df(n: int = 100) -> pd.DataFrame:
    """Synthetic OHLC dataframe with trending close."""
    np.random.seed(42)
    closes = 1.30 + np.cumsum(np.random.randn(n) * 0.0003)
    highs = closes + np.abs(np.random.randn(n)) * 0.0005
    lows = closes - np.abs(np.random.randn(n)) * 0.0005
    opens = closes + np.random.randn(n) * 0.0002
    return pd.DataFrame({
        "open": opens, "high": highs, "low": lows, "close": closes,
        "volume": np.ones(n),
    }, index=pd.date_range("2026-01-01", periods=n, freq="15min", tz="UTC"))


def test_rsi_returns_series_in_range():
    df = _toy_df(100)
    rsi = _rsi(df["close"], 14).dropna()
    assert (rsi >= 0).all() and (rsi <= 100).all()


def test_atr_positive():
    df = _toy_df(100)
    atr = _atr(df, 14).dropna()
    assert (atr > 0).all()


def test_ema_smooths():
    df = _toy_df(100)
    ema = _ema(df["close"], 10)
    # EMA should have less variance than raw close
    assert ema.std() < df["close"].std() * 1.05


def test_adx_in_range():
    df = _toy_df(100)
    adx = _adx(df, 14).dropna()
    # ADX is typically 0-100
    assert (adx >= 0).all()
    assert (adx <= 100).all()


# ── Technical features ──────────────────────────────────────────────────────

def test_technical_features_handles_empty():
    out = _technical_features("USDCAD", "long",
                              pd.DataFrame(), None, None)
    assert "rsi_at_entry" in out
    assert out["rsi_at_entry"] == 50.0  # default


def test_technical_features_returns_all_keys():
    df = _toy_df(100)
    out = _technical_features("USDCAD", "long", df, df, df)
    expected = {
        "rsi_at_entry", "rsi_depth", "atr_pctl_60d", "adx_h4",
        "h4_trend_score", "h1_trend_score", "m15_atr_norm",
        "adr_pct_used", "bars_since_extreme", "range_compress",
    }
    assert set(out.keys()) == expected


def test_technical_features_rsi_depth_long():
    """For long signals, rsi_depth = |rsi - 38|."""
    df = _toy_df(100)
    out = _technical_features("USDCAD", "long", df, None, None)
    assert out["rsi_depth"] == abs(out["rsi_at_entry"] - 38)


def test_technical_features_rsi_depth_short():
    df = _toy_df(100)
    out = _technical_features("USDCAD", "short", df, None, None)
    assert out["rsi_depth"] == abs(out["rsi_at_entry"] - 62)


def test_bars_since_extreme_in_range():
    df = _toy_df(100)
    out = _technical_features("USDCAD", "long", df, None, None)
    assert 0 <= out["bars_since_extreme"] <= 19


# ── Strategy state ──────────────────────────────────────────────────────────

def test_strategy_state_no_history():
    """Empty journal returns defaults."""
    out = _strategy_state_features(
        "USDCAD", "TREND_RSI",
        datetime(2026, 6, 5, tzinfo=timezone.utc),
        lambda strategy, before, limit: [],
    )
    assert out["strat_wr_last_20"] == 0.5
    assert out["strat_streak_signed"] == 0


def test_strategy_state_computes_wr_and_streak():
    """Mocked journal: 3 wins then 2 losses → most recent streak is 2 losses."""
    rows = [
        {"pair": "USDCAD", "outcome": "loss", "r_multiple": -1.0,
         "ts": "2026-06-04T00:00:00+00:00"},
        {"pair": "USDCAD", "outcome": "loss", "r_multiple": -1.0,
         "ts": "2026-06-03T00:00:00+00:00"},
        {"pair": "USDCAD", "outcome": "win", "r_multiple": 2.5,
         "ts": "2026-06-02T00:00:00+00:00"},
        {"pair": "USDCAD", "outcome": "win", "r_multiple": 2.5,
         "ts": "2026-06-01T00:00:00+00:00"},
        {"pair": "USDCAD", "outcome": "win", "r_multiple": 2.5,
         "ts": "2026-05-31T00:00:00+00:00"},
    ]
    out = _strategy_state_features(
        "USDCAD", "TREND_RSI",
        datetime(2026, 6, 5, tzinfo=timezone.utc),
        lambda strategy, before, limit: rows,
    )
    assert out["strat_wr_last_20"] == 0.6  # 3/5
    assert out["strat_streak_signed"] == -2  # 2 losses
    assert out["strat_net_r_last_20"] == pytest.approx(5.5)


def test_strategy_state_pair_specific_wr():
    rows = [
        {"pair": "USDCAD", "outcome": "win", "r_multiple": 2.5,
         "ts": "2026-06-04T00:00:00+00:00"},
        {"pair": "EURUSD", "outcome": "loss", "r_multiple": -1.0,
         "ts": "2026-06-03T00:00:00+00:00"},
    ]
    out = _strategy_state_features(
        "USDCAD", "TREND_RSI",
        datetime(2026, 6, 5, tzinfo=timezone.utc),
        lambda strategy, before, limit: rows,
    )
    # Only the USDCAD win counts for pair_strat_wr
    assert out["pair_strat_wr_last_10"] == 1.0


# ── LLM features ────────────────────────────────────────────────────────────

def test_llm_features_none():
    out = _llm_features(None)
    assert out["llm_confidence"] == 0.5
    assert out["llm_decision_take"] == 0


def test_llm_features_take():
    class R:
        confidence = 0.85
        decision = "TAKE"
    out = _llm_features(R())
    assert out["llm_decision_take"] == 1
    assert out["llm_decision_skip"] == 0
    assert out["llm_confidence"] == 0.85


def test_llm_features_skip():
    class R:
        confidence = 0.9
        decision = "SKIP"
    out = _llm_features(R())
    assert out["llm_decision_skip"] == 1
    assert out["llm_decision_take"] == 0


def test_llm_features_reduce():
    class R:
        confidence = 0.6
        decision = "REDUCE"
    out = _llm_features(R())
    assert out["llm_decision_reduce"] == 1


# ── End-to-end FeatureExtractor ─────────────────────────────────────────────

def test_extractor_returns_28_fields():
    df = _toy_df(100)
    fe = FeatureExtractor()
    fs = fe.extract(
        pair="USDCAD", strategy="TREND_RSI", direction="long",
        ts=datetime(2026, 6, 5, 14, 0, tzinfo=timezone.utc),
        m15=df, h1=df, h4=df,
        llm_result=None,
        journal_query=lambda strategy, before, limit: [],
    )
    d = fs.to_dict()
    assert len(d) == 28
    assert set(d.keys()) == set(FEATURE_NAMES)


def test_extractor_categorical_correct():
    df = _toy_df(100)
    fe = FeatureExtractor()
    fs = fe.extract(
        pair="EURUSD", strategy="ASIA_HL", direction="short",
        ts=datetime(2026, 6, 5, 14, 0, tzinfo=timezone.utc),  # Friday NY-AM
        m15=df, h1=df, h4=df,
        journal_query=lambda strategy, before, limit: [],
    )
    assert fs.strategy == "ASIA_HL"
    assert fs.direction == "short"
    assert fs.pair == "EURUSD"
    assert fs.session == "NY-AM"
    assert fs.dow == "Fri"


def test_extractor_naive_timestamp_normalised():
    df = _toy_df(100)
    fe = FeatureExtractor()
    fs = fe.extract(
        pair="USDCAD", strategy="TREND_RSI", direction="long",
        ts=datetime(2026, 6, 5, 14, 0),  # naive — should become UTC
        m15=df, h1=df, h4=df,
        journal_query=lambda strategy, before, limit: [],
    )
    assert fs.session == "NY-AM"


def test_extractor_handles_missing_dataframes():
    """Missing h1/h4 should not crash — returns defaults."""
    df = _toy_df(50)  # short m15 only
    fe = FeatureExtractor()
    fs = fe.extract(
        pair="USDCAD", strategy="TREND_RSI", direction="long",
        ts=datetime(2026, 6, 5, 14, 0, tzinfo=timezone.utc),
        m15=df, h1=None, h4=None,
        journal_query=lambda strategy, before, limit: [],
    )
    assert fs.h4_trend_score == 0.0  # default
    assert fs.h1_trend_score == 0.0


def test_extractor_to_dict_jsonable():
    """Output must round-trip through json.dumps."""
    import json
    df = _toy_df(100)
    fe = FeatureExtractor()
    fs = fe.extract(
        pair="USDCAD", strategy="TREND_RSI", direction="long",
        ts=datetime(2026, 6, 5, 14, 0, tzinfo=timezone.utc),
        m15=df, h1=df, h4=df,
        journal_query=lambda strategy, before, limit: [],
    )
    serialised = json.dumps(fs.to_dict(), default=str)
    parsed = json.loads(serialised)
    assert parsed["pair"] == "USDCAD"


# ── DB integration: features_json round-trip ────────────────────────────────

def test_db_update_features_persists(tmp_path, monkeypatch):
    """update_features() should write JSON we can read back."""
    monkeypatch.setenv("FSP_DATA_DIR", str(tmp_path))
    monkeypatch.setattr("fsp.journal.db.DB_PATH", tmp_path / "journal.db")
    from fsp.journal.db import (
        conn, migrate, log_intraday_signal, update_features,
    )
    from fsp.signals.base import Signal

    # Insert a fake signal
    sig = Signal(
        pair="USDCAD", strategy="TREND_RSI", direction="long",
        entry=1.38, sl=1.378, tp1=1.385, tp2=1.39,
        inv_pips=20.0, rr_tp1=2.5, rr_tp2=5.0, risk_r=1.0,
        note="test", ts="2026-06-05T14:00:00+00:00",
        context={},
    )
    sig_id = log_intraday_signal(sig, "TEST|USDCAD|TREND_RSI|long", sent=True)
    assert sig_id > 0

    update_features(sig_id, {"strategy": "TREND_RSI", "rsi_at_entry": 35.5})

    with conn() as c:
        migrate(c)
        row = c.execute(
            "SELECT features_json FROM intraday_signals WHERE id=?",
            (sig_id,),
        ).fetchone()
    assert row[0] is not None
    import json
    parsed = json.loads(row[0])
    assert parsed["rsi_at_entry"] == 35.5
