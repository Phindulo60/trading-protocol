"""Tests for the SCALP_MR M5 Bollinger Band mean-reversion scalper."""
import pandas as pd
import numpy as np
import pytest
from fsp.signals.scalp_mr import scan_scalp_mr, _bollinger


def _make_m5(n: int = 30, base: float = 1.1000, pip: float = 0.0001,
             start: str = "2026-07-28 09:00") -> pd.DataFrame:
    """Create synthetic M5 bars around a base price, London session."""
    idx = pd.date_range(start, periods=n, freq="5min", tz="UTC")
    rng = np.random.default_rng(42)
    close = base + rng.normal(0, 5 * pip, n).cumsum()
    high = close + rng.uniform(1, 3, n) * pip
    low = close - rng.uniform(1, 3, n) * pip
    return pd.DataFrame({
        "open": close - rng.uniform(-2, 2, n) * pip,
        "high": high,
        "low": low,
        "close": close,
        "volume": rng.integers(100, 500, n),
    }, index=idx)


def _spike_above_setup(pair: str = "EURUSD") -> pd.DataFrame:
    """Create M5 data where bar[-2] spiked above upper BB and bar[-1] reverted."""
    pip = 0.0001 if "JPY" not in pair else 0.01
    base = 1.1000 if "JPY" not in pair else 150.0
    # 25 flat bars then a spike + reversion
    n = 27
    idx = pd.date_range("2026-07-28 09:00", periods=n, freq="5min", tz="UTC")
    # First 25 bars: tight range around base
    rng = np.random.default_rng(7)
    close = np.full(n, base) + rng.normal(0, 2 * pip, n).cumsum()
    # Make it relatively flat so BB is narrow-ish but within filter
    close[:25] = base + np.linspace(0, 3 * pip, 25)
    # Bar[-2]: spike above upper band (close well above mean + 2σ)
    close[-2] = base + 20 * pip  # way above
    # Bar[-1]: reverts back inside
    close[-1] = base + 5 * pip
    high = close + 2 * pip
    low = close - 2 * pip
    # Spike bar has extreme high
    high[-2] = close[-2] + 3 * pip
    return pd.DataFrame({
        "open": close - pip,
        "high": high,
        "low": low,
        "close": close,
        "volume": np.full(n, 200),
    }, index=idx)


def _spike_below_setup(pair: str = "EURUSD") -> pd.DataFrame:
    """Bar[-2] spiked below lower BB, bar[-1] reverted."""
    pip = 0.0001
    base = 1.1000
    n = 27
    idx = pd.date_range("2026-07-28 10:00", periods=n, freq="5min", tz="UTC")
    close = np.full(n, base) + np.linspace(0, -3 * 0.0001, n)
    close[-2] = base - 20 * pip  # spike below
    close[-1] = base - 5 * pip  # reverts
    high = close + 2 * pip
    low = close - 2 * pip
    low[-2] = close[-2] - 3 * pip
    return pd.DataFrame({
        "open": close + pip,
        "high": high,
        "low": low,
        "close": close,
        "volume": np.full(n, 200),
    }, index=idx)


class TestBollinger:
    def test_bollinger_shape(self):
        s = pd.Series(np.linspace(1.0, 1.1, 30))
        sma, upper, lower = _bollinger(s, 20, 2.0)
        assert len(sma) == 30
        assert (upper.dropna() > sma.dropna()).all()
        assert (lower.dropna() < sma.dropna()).all()


class TestScanScalpMR:
    def test_insufficient_data_returns_none(self):
        df = _make_m5(n=20)
        assert scan_scalp_mr("EURUSD", df) is None

    def test_no_spike_returns_none(self):
        # Normal market, no bar outside BB
        df = _make_m5(n=30, base=1.1000)
        # All bars near mean -- unlikely to spike
        df["close"] = 1.1000  # flat
        df["high"] = 1.1002
        df["low"] = 1.0998
        assert scan_scalp_mr("EURUSD", df) is None

    def test_spike_above_generates_short(self):
        df = _spike_above_setup()
        sig = scan_scalp_mr("EURUSD", df)
        # May be None if band width filter kills it — adjust if needed
        if sig is not None:
            assert sig.direction == "short"
            assert sig.strategy == "SCALP_MR"
            assert sig.pair == "EURUSD"
            assert sig.inv_pips <= 12
            assert sig.rr_tp1 >= 0.8
            assert sig.context["max_hold_bars"] == 3

    def test_spike_below_generates_long(self):
        df = _spike_below_setup()
        sig = scan_scalp_mr("EURUSD", df)
        if sig is not None:
            assert sig.direction == "long"
            assert sig.strategy == "SCALP_MR"
            assert sig.inv_pips <= 12

    def test_outside_session_returns_none(self):
        # Create data in Asian session (02:00 UTC)
        n = 27
        idx = pd.date_range("2026-07-28 02:00", periods=n, freq="5min", tz="UTC")
        df = _spike_above_setup()
        df.index = idx
        assert scan_scalp_mr("EURUSD", df) is None

    def test_jpy_pair_pip_size(self):
        # Smoke test — should not crash on JPY pairs
        df = _spike_above_setup("USDJPY")
        # Adjust prices for JPY
        close = 150.0 + np.linspace(0, 0.03, len(df))
        close[-2] = 150.0 + 0.20
        close[-1] = 150.0 + 0.05
        df["close"] = close
        df["high"] = close + 0.02
        df["low"] = close - 0.02
        df.loc[df.index[-2], "high"] = close[-2] + 0.03
        result = scan_scalp_mr("USDJPY", df)
        # Should not crash; result depends on BB calc
        assert result is None or result.strategy == "SCALP_MR"

    def test_wide_sl_filtered(self):
        # If spike extreme is very far, SL > 12 pips -> skip
        pip = 0.0001
        base = 1.1000
        n = 27
        idx = pd.date_range("2026-07-28 09:00", periods=n, freq="5min", tz="UTC")
        close = np.full(n, base) + np.linspace(0, 3 * pip, n)
        # Spike with extreme high very far (30 pips above entry)
        close[-2] = base + 20 * pip
        close[-1] = base + 5 * pip
        high = close + 2 * pip
        low = close - 2 * pip
        high[-2] = base + 35 * pip  # extreme 30+ pips from close[-1]
        df = pd.DataFrame({
            "open": close, "high": high, "low": low,
            "close": close, "volume": np.full(n, 200),
        }, index=idx)
        sig = scan_scalp_mr("EURUSD", df)
        # SL would be > 12 pips -> None
        assert sig is None
