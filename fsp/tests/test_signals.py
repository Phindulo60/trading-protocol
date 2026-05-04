"""Unit tests for ECM and ARB signal strategies."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fsp.signals.momentum import scan_momentum, _ema, _rsi
from fsp.signals.breakout import scan_breakout, _asian_range
from fsp.signals.base import Signal


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_df(n: int = 100, start: str = "2024-01-15 07:00:00",
             freq: str = "15min",
             base: float = 1.10000,
             trend: float = 0.0,
             vol: float = 0.0002) -> pd.DataFrame:
    """Synthetic OHLCV bars with optional trend."""
    idx = pd.date_range(start, periods=n, freq=freq, tz="UTC")
    np.random.seed(42)
    mid = base + np.cumsum(np.random.randn(n) * vol + trend)
    rng = np.abs(np.random.randn(n)) * vol * 0.5 + vol * 0.2
    return pd.DataFrame({
        "open":   mid - rng * 0.3,
        "high":   mid + rng * 0.7,
        "low":    mid - rng * 0.7,
        "close":  mid + rng * 0.3,
        "volume": np.ones(n) * 1000,
    }, index=idx)


def _make_daily(n: int = 20) -> pd.DataFrame:
    return _make_df(n, start="2024-01-01", freq="1D", base=1.10, vol=0.005)


# ── EMA helpers ───────────────────────────────────────────────────────────────

class TestEMAHelpers:
    def test_ema_converges(self):
        s = pd.Series([1.0] * 100)
        assert abs(_ema(s, 8).iloc[-1] - 1.0) < 1e-9

    def test_rsi_neutral_random_walk(self):
        np.random.seed(0)
        s = pd.Series(1.1 + np.cumsum(np.random.randn(200) * 0.0001))
        rsi = _rsi(s, 14)
        # RSI of a random walk should hover near 50
        assert 35 < rsi.iloc[-1] < 65

    def test_rsi_always_bullish(self):
        # Monotonically rising → RSI near 100
        s = pd.Series(np.linspace(1.0, 2.0, 60))
        rsi = _rsi(s, 14)
        assert rsi.iloc[-1] > 80

    def test_rsi_always_bearish(self):
        s = pd.Series(np.linspace(2.0, 1.0, 60))
        rsi = _rsi(s, 14)
        assert rsi.iloc[-1] < 20


# ── Momentum scan ─────────────────────────────────────────────────────────────

class TestMomentumScan:
    def test_no_signal_on_tiny_df(self):
        tiny = _make_df(10)
        h1 = _make_df(10, freq="60min")
        d = _make_daily()
        assert scan_momentum("EURUSD", tiny, h1, d) is None

    def test_bull_cross_during_london(self):
        """Construct a scenario where EMA8 just crossed above EMA21 during London session."""
        # Build M15 bars where close rises sharply in last few bars to force a bull cross
        n = 80
        # First 75 bars: sideways/down slightly so EMA8 < EMA21
        # Last 5 bars: sharp up to flip the cross
        np.random.seed(1)
        closes_down = 1.10 + np.cumsum(np.random.randn(75) * 0.0001 - 0.00005)
        closes_up = closes_down[-1] + np.cumsum(np.ones(5) * 0.0008)
        closes = np.concatenate([closes_down, closes_up])
        highs = closes + 0.0003
        lows = closes - 0.0003
        opens = np.roll(closes, 1)
        opens[0] = closes[0]

        # Set time to London session (08:00 UTC = 03:00 NY)
        idx = pd.date_range("2024-01-15 08:00:00", periods=n, freq="15min", tz="UTC")
        m15 = pd.DataFrame({
            "open": opens, "high": highs, "low": lows,
            "close": closes, "volume": np.ones(n) * 1000
        }, index=idx)

        # H1 bars also trending up
        h1_n = 70
        h1_idx = pd.date_range("2024-01-12 00:00:00", periods=h1_n, freq="60min", tz="UTC")
        h1_closes = 1.10 + np.cumsum(np.ones(h1_n) * 0.0002)
        h1 = pd.DataFrame({
            "open": h1_closes - 0.0002, "high": h1_closes + 0.0003,
            "low": h1_closes - 0.0003, "close": h1_closes,
            "volume": np.ones(h1_n) * 1000
        }, index=h1_idx)

        d = _make_daily()
        sig = scan_momentum("EURUSD", m15, h1, d)
        if sig is not None:
            assert sig.strategy == "ECM"
            assert sig.direction == "long"
            assert sig.entry > sig.sl
            assert sig.tp1 > sig.entry
            assert sig.rr_tp1 == pytest.approx(1.5)
            assert sig.inv_pips <= 25

    def test_skip_outside_session(self):
        """Signal should be None if last bar is in Asia/OFF session (02:00 UTC = 21:00 NY)."""
        n = 80
        np.random.seed(2)
        closes_down = 1.10 + np.cumsum(np.random.randn(75) * 0.0001 - 0.00005)
        closes_up = closes_down[-1] + np.cumsum(np.ones(5) * 0.0008)
        closes = np.concatenate([closes_down, closes_up])
        highs = closes + 0.0003
        lows = closes - 0.0003
        opens = np.roll(closes, 1); opens[0] = closes[0]

        # Asia session: 02:00 UTC (= 21:00 NY previous day)
        idx = pd.date_range("2024-01-15 02:00:00", periods=n, freq="15min", tz="UTC")
        m15 = pd.DataFrame({"open": opens, "high": highs, "low": lows,
                             "close": closes, "volume": np.ones(n)}, index=idx)
        h1 = _make_df(70, freq="60min", trend=0.0002)
        d = _make_daily()
        # Outside allowed sessions → should return None
        assert scan_momentum("EURUSD", m15, h1, d) is None

    def test_skip_with_overbought_rsi(self):
        """If RSI > 68 for a long, should return None."""
        n = 80
        # Pure monotonic rise → RSI → 100
        closes = 1.10 + np.linspace(0, 0.10, n)
        highs = closes + 0.0002
        lows = closes - 0.0002
        opens = np.roll(closes, 1); opens[0] = closes[0]
        idx = pd.date_range("2024-01-15 08:00:00", periods=n, freq="15min", tz="UTC")
        m15 = pd.DataFrame({"open": opens, "high": highs, "low": lows,
                             "close": closes, "volume": np.ones(n)}, index=idx)
        h1 = _make_df(70, freq="60min", trend=0.0002)
        d = _make_daily()
        # EMA8 will be above EMA21 throughout (no fresh cross), and RSI > 68
        assert scan_momentum("EURUSD", m15, h1, d) is None


# ── Asian range helper ────────────────────────────────────────────────────────

class TestAsianRange:
    def _make_m5_with_asian(self, asian_high: float, asian_low: float,
                             breakout_close: float | None = None,
                             breakout_ts: str = "2024-01-15 07:10:00") -> pd.DataFrame:
        """Create M5 bars with a defined Asian session range and optional breakout bar."""
        rows = []
        # Asian window bars (00:00–07:00 UTC)
        for h in range(7):
            for m in range(0, 60, 5):
                ts = pd.Timestamp(f"2024-01-15 {h:02d}:{m:02d}:00", tz="UTC")
                mid = (asian_high + asian_low) / 2
                rows.append({"ts": ts, "open": mid, "high": asian_high if h == 2 and m == 0 else mid + 0.0001,
                              "low": asian_low if h == 4 and m == 0 else mid - 0.0001,
                              "close": mid, "volume": 1000})
        # Breakout bar if specified
        if breakout_close is not None:
            ts = pd.Timestamp(breakout_ts, tz="UTC")
            rows.append({"ts": ts, "open": asian_high,
                         "high": breakout_close + 0.0002,
                         "low": asian_high - 0.0001,
                         "close": breakout_close, "volume": 1000})
        df = pd.DataFrame(rows).set_index("ts")
        return df

    def test_asian_range_detected(self):
        # Include a breakout bar AFTER 07:00 UTC so _asian_range sees today's window
        df = self._make_m5_with_asian(1.1050, 1.1020,
                                       breakout_close=1.1053,
                                       breakout_ts="2024-01-15 07:05:00")
        result = _asian_range(df)
        assert result is not None
        ah, al = result
        assert ah == pytest.approx(1.1050)
        assert al == pytest.approx(1.1020)

    def test_range_too_small_rejected(self):
        # 5 pip range → below 15 pip threshold
        df = self._make_m5_with_asian(1.1005, 1.1000, breakout_close=1.1008,
                                       breakout_ts="2024-01-15 07:05:00")
        h1 = _make_df(60, freq="60min")
        d = _make_daily()
        assert scan_breakout("EURUSD", df, h1, d) is None

    def test_range_too_large_rejected(self):
        # 80 pip range → above 60 pip threshold
        df = self._make_m5_with_asian(1.1080, 1.1000, breakout_close=1.1085,
                                       breakout_ts="2024-01-15 07:05:00")
        h1 = _make_df(60, freq="60min")
        d = _make_daily()
        assert scan_breakout("EURUSD", df, h1, d) is None

    def test_bull_breakout_signal(self):
        """Valid bull breakout: 30-pip range, close above asian_high during LO."""
        asian_high, asian_low = 1.1030, 1.1000  # 30 pips
        df = self._make_m5_with_asian(asian_high, asian_low,
                                       breakout_close=asian_high + 0.0003,  # 3 pips above
                                       breakout_ts="2024-01-15 07:15:00")
        h1 = _make_df(60, freq="60min", base=1.10)
        d = _make_daily()
        sig = scan_breakout("EURUSD", df, h1, d)
        if sig is not None:
            assert sig.strategy == "ARB"
            assert sig.direction == "long"
            assert sig.entry > asian_high
            assert sig.sl < asian_high          # SL below the broken level
            assert sig.tp1 > sig.entry
            assert sig.rr_tp1 >= 1.5
            assert sig.inv_pips <= 20

    def test_outside_breakout_window_rejected(self):
        """No signal if current bar is outside 07:00–09:30 UTC."""
        asian_high, asian_low = 1.1030, 1.1000
        # Bar at 11:00 UTC — outside breakout window
        df = self._make_m5_with_asian(asian_high, asian_low,
                                       breakout_close=asian_high + 0.0003,
                                       breakout_ts="2024-01-15 11:00:00")
        h1 = _make_df(60, freq="60min")
        d = _make_daily()
        assert scan_breakout("EURUSD", df, h1, d) is None


# ── Signal dataclass ──────────────────────────────────────────────────────────

class TestSignalBase:
    def test_dedup_key_stable(self):
        sig = Signal(
            strategy="ECM", pair="EURUSD", direction="long",
            entry=1.10500, sl=1.10300, tp1=1.10800, tp2=1.11000,
            inv_pips=20.0, rr_tp1=1.5, rr_tp2=2.5,
            risk_r=1.0, note="test", ts="2024-01-15T08:00:00+00:00",
        )
        assert sig.dedup_key == "EURUSD|ECM|long|1.10500"

    def test_dedup_key_changes_on_direction(self):
        sig1 = Signal("ECM", "EURUSD", "long", 1.1050, 1.1030, 1.1080, None,
                      20.0, 1.5, None, 1.0, "", "2024-01-15T08:00:00+00:00")
        sig2 = Signal("ECM", "EURUSD", "short", 1.1050, 1.1070, 1.1020, None,
                      20.0, 1.5, None, 1.0, "", "2024-01-15T08:00:00+00:00")
        assert sig1.dedup_key != sig2.dedup_key


# ── TREND_RSI ────────────────────────────────────────────────────────────────

class TestTrendRSI:
    """Tests for the backtested TREND_RSI strategy."""

    def _make_h4(self, n: int = 30, trend: float = 0.0002) -> pd.DataFrame:
        """H4 bars in a clear trend so EMA20 is meaningfully directional."""
        idx = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC")
        closes = 1.10 + np.cumsum(np.ones(n) * trend)
        return pd.DataFrame({
            "open": closes - 0.0005, "high": closes + 0.001,
            "low":  closes - 0.001,  "close": closes,
            "volume": np.ones(n) * 1000,
        }, index=idx)

    def test_returns_none_on_small_df(self):
        from fsp.signals.alpha import scan_trend_rsi
        tiny = _make_df(10)
        h4 = self._make_h4(5)
        assert scan_trend_rsi("EURUSD", tiny, h4) is None

    def test_long_signal_during_ny_am(self):
        """H4 bull + RSI<38 during NY-AM → should fire LONG."""
        from fsp.signals.alpha import scan_trend_rsi
        n = 80
        # H4 uptrend (EMA20 bullish)
        h4 = self._make_h4(30, trend=0.0003)

        # M15 bars with RSI < 38: build a declining close sequence for last 20 bars
        np.random.seed(7)
        closes_base = 1.10 + np.cumsum(np.random.randn(60) * 0.0002)
        # Decline sharply in last 20 bars to drive RSI below 38
        closes_down = closes_base[-1] - np.cumsum(np.ones(20) * 0.0004)
        closes = np.concatenate([closes_base, closes_down])
        highs = closes + 0.0003; lows = closes - 0.0003
        opens = np.roll(closes, 1); opens[0] = closes[0]
        # Place last bar in NY-AM (13:00 UTC = 09:00 NY)
        idx = pd.date_range("2024-01-15 06:00:00", periods=len(closes),
                             freq="15min", tz="UTC")
        m15 = pd.DataFrame({"open": opens, "high": highs, "low": lows,
                              "close": closes, "volume": np.ones(len(closes))},
                             index=idx)
        sig = scan_trend_rsi("EURUSD", m15, h4)
        if sig is not None:
            assert sig.strategy == "TREND_RSI"
            assert sig.direction == "long"
            assert sig.entry > sig.sl
            assert sig.tp1 > sig.entry
            assert sig.rr_tp1 == pytest.approx(3.5)

    def test_no_signal_outside_ny(self):
        """Signal should not fire during LONDON session."""
        from fsp.signals.alpha import scan_trend_rsi
        h4 = self._make_h4(30, trend=0.0003)
        n = 80
        np.random.seed(8)
        closes = 1.10 - np.cumsum(np.ones(n) * 0.0003)  # declining → RSI low
        highs = closes + 0.0003; lows = closes - 0.0003
        opens = np.roll(closes, 1); opens[0] = closes[0]
        # London session: 07:00-12:00 UTC (02:00-07:00 NY)
        idx = pd.date_range("2024-01-15 08:00:00", periods=n, freq="15min", tz="UTC")
        m15 = pd.DataFrame({"open": opens, "high": highs, "low": lows,
                              "close": closes, "volume": np.ones(n)}, index=idx)
        assert scan_trend_rsi("EURUSD", m15, h4) is None

    def test_no_signal_on_friday(self):
        """Signal should not fire on Friday."""
        from fsp.signals.alpha import scan_trend_rsi
        h4 = self._make_h4(30, trend=0.0003)
        n = 80
        closes = 1.10 - np.cumsum(np.ones(n) * 0.0003)
        highs = closes + 0.0003; lows = closes - 0.0003
        opens = np.roll(closes, 1); opens[0] = closes[0]
        # Friday NY-AM: 2024-01-19 is a Friday
        idx = pd.date_range("2024-01-19 13:00:00", periods=n, freq="15min", tz="UTC")
        m15 = pd.DataFrame({"open": opens, "high": highs, "low": lows,
                              "close": closes, "volume": np.ones(n)}, index=idx)
        assert scan_trend_rsi("EURUSD", m15, h4) is None

    def test_short_signal_h4_bear(self):
        """H4 bear + RSI>62 → should fire SHORT."""
        from fsp.signals.alpha import scan_trend_rsi
        # H4 downtrend
        h4 = self._make_h4(30, trend=-0.0003)
        n = 80
        # Rising close → RSI > 62
        closes = 1.10 + np.cumsum(np.ones(n) * 0.0004)
        highs = closes + 0.0003; lows = closes - 0.0003
        opens = np.roll(closes, 1); opens[0] = closes[0]
        idx = pd.date_range("2024-01-15 13:00:00", periods=n, freq="15min", tz="UTC")
        m15 = pd.DataFrame({"open": opens, "high": highs, "low": lows,
                              "close": closes, "volume": np.ones(n)}, index=idx)
        sig = scan_trend_rsi("EURUSD", m15, h4)
        if sig is not None:
            assert sig.direction == "short"
            assert sig.sl > sig.entry
            assert sig.tp1 < sig.entry
