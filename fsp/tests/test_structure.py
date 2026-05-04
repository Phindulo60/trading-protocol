from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from fsp.structure.swings import find_swings
from fsp.structure.fvg import find_fvgs


def _mk_df(highs, lows, opens=None, closes=None):
    n = len(highs)
    ts = [datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(hours=i) for i in range(n)]
    opens = opens or [(h + l) / 2 for h, l in zip(highs, lows)]
    closes = closes or opens
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": [0] * n},
        index=pd.DatetimeIndex(ts, name="ts"),
    )


def test_swings_basic():
    # peak at index 5, trough at index 11
    highs = [10, 11, 12, 13, 14, 20, 13, 12, 11, 10, 9, 5, 9, 10, 11, 12]
    lows = [h - 1 for h in highs]
    df = _mk_df(highs, lows)
    sw = find_swings(df, length=3)
    kinds = [(s.kind, round(s.price, 2)) for s in sw]
    assert ("high", 20.0) in kinds
    assert ("low", 4.0) in kinds


def test_bull_fvg():
    # candle i-2 high = 10, candle i low = 12 → bullish FVG [10, 12]
    highs = [10, 11, 11, 13, 14]
    lows = [9, 10, 12, 12, 13]
    df = _mk_df(highs, lows)
    fvgs = find_fvgs(df, tf="H1")
    bull = [f for f in fvgs if f.direction == "bull"]
    assert bull
    assert round(bull[0].bottom, 2) == 10.0
    assert round(bull[0].top, 2) == 12.0


def test_bear_fvg():
    highs = [14, 13, 10, 10, 9]
    lows = [13, 12, 9, 9, 8]
    df = _mk_df(highs, lows)
    fvgs = find_fvgs(df, tf="H1")
    bear = [f for f in fvgs if f.direction == "bear"]
    assert bear


from fsp.structure.displacement import find_displacements
from fsp.structure.order_blocks import find_order_blocks


def test_displacement_up():
    # build 21 quiet bars then one huge up-bar
    n = 25
    highs = [10.1] * n
    lows = [9.9] * n
    opens = [10.0] * n
    closes = [10.0] * n
    # displacement at i=22
    i = 22
    opens[i] = 10.0
    closes[i] = 12.0
    highs[i] = 12.1
    lows[i] = 9.99
    df = _mk_df(highs, lows, opens, closes)
    disp = find_displacements(df, mult=1.5, length=20)
    assert any(d.direction == "up" for d in disp)


def test_order_block_bullish():
    n = 25
    highs = [10.1] * n
    lows = [9.9] * n
    opens = [10.0] * n
    closes = [10.0] * n
    # bar 21 = bearish candle, bar 22 = bullish displacement
    opens[21], closes[21], highs[21], lows[21] = 10.05, 9.95, 10.08, 9.92
    opens[22], closes[22], highs[22], lows[22] = 9.95, 12.0, 12.1, 9.94
    df = _mk_df(highs, lows, opens, closes)
    obs = find_order_blocks(df, tf="H1", mult=1.5, length=20)
    assert any(o.direction == "bull" for o in obs)


from fsp.context.cycle import classify_cycle
from fsp.context.bias import compute_of_bias
from fsp.data.types import Cycle, OFBias


def test_cycle_neutral_on_small():
    df = _mk_df([10.1] * 10, [9.9] * 10)
    cs = classify_cycle(df)
    assert cs.cycle == Cycle.NEUTRAL


def test_bias_bullish_trend():
    # simple uptrend: each swing higher than the last
    highs, lows = [], []
    base = 100.0
    for i in range(100):
        # rising zigzag
        amp = 1.0
        base += 0.2
        highs.append(base + amp)
        lows.append(base - amp)
    # carve swings by modulating
    for i in range(5, 100, 10):
        highs[i] += 2
    for i in range(10, 100, 10):
        lows[i] -= 2
    df = _mk_df(highs, lows)
    b = compute_of_bias(df, length=3)
    # at least should not be BEAR
    assert b.bias != OFBias.BEAR
