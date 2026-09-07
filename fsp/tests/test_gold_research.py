"""Guards for the gold research harness: no look-ahead, correct stats."""
import numpy as np
import pandas as pd
import pytest

from fsp.research.gold.evaluate import nw_tstat, walk_forward, ZeroForecaster, RidgeForecaster
from fsp.research.gold.features import build_features, forward_return, triple_barrier, attach_macro


def _synthetic(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="h", tz="UTC")
    r = rng.normal(0, 0.002, n)
    c = 1800 * np.exp(np.cumsum(r))
    hi = c * (1 + np.abs(rng.normal(0, 0.001, n)))
    lo = c * (1 - np.abs(rng.normal(0, 0.001, n)))
    return pd.DataFrame({"open": np.roll(c, 1), "high": hi, "low": lo, "close": c,
                         "volume": rng.integers(100, 1000, n), "spread": 0.3}, index=idx)


def test_nw_tstat_iid_matches_classic():
    x = np.random.default_rng(1).normal(0.5, 1, 5000)
    classic = x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))
    assert abs(nw_tstat(x, 0) - classic) < 0.02


def test_nw_tstat_deflates_overlapping_series():
    rng = np.random.default_rng(2)
    e = rng.normal(0, 1, 20000)
    h = 24
    overlapped = np.convolve(e, np.ones(h), mode="valid") + 0.05  # h-bar rolling sums
    naive = overlapped.mean() / (overlapped.std(ddof=1) / np.sqrt(len(overlapped)))
    nw = nw_tstat(overlapped, h - 1)
    assert nw < naive / 3  # sqrt(24) ~ 4.9x inflation


def test_features_have_no_lookahead():
    df = _synthetic()
    f_full = build_features(df, 24)
    f_cut = build_features(df.iloc[:2000], 24)
    common = f_cut.index
    pd.testing.assert_frame_equal(f_full.loc[common], f_cut, check_dtype=False)


def test_forward_return_is_future():
    df = _synthetic()
    y = forward_return(df, 5)
    assert np.isclose(y.iloc[0], np.log(df.close.iloc[5] / df.close.iloc[0]))
    assert y.iloc[-5:].isna().all()


def test_triple_barrier_labels():
    df = _synthetic()
    vol = pd.Series(0.002, index=df.index)
    tb = triple_barrier(df, h=24, tp_mult=2, sl_mult=2, vol=vol)
    assert set(tb.label.unique()) <= {-1, 0, 1}
    assert (tb.bars <= 24).all()
    # tp hit -> positive ret, sl hit -> negative
    assert (tb.ret[tb.label == 1] > 0).all() and (tb.ret[tb.label == -1] < 0).all()


def test_attach_macro_lags_one_day():
    df = _synthetic(n=24 * 40)
    f = build_features(df, 24)
    days = pd.date_range("2019-12-01", "2020-03-01", freq="D", tz="UTC")
    macro = pd.DataFrame({"dxy": np.arange(len(days), dtype=float) + 100}, index=days)
    out = attach_macro(f, macro)
    # bar at 2020-01-10 10:00 must see the change dated 2020-01-09, not 2020-01-10
    ts = pd.Timestamp("2020-01-10 10:00", tz="UTC")
    expected = np.log(macro.dxy).diff().loc["2020-01-09"]
    assert np.isclose(out.loc[ts, "dxy_chg1"], expected)


def test_walk_forward_purges_and_covers_oos():
    df = _synthetic()
    f = build_features(df, 24).drop(columns=["hour", "dow", "dom"])
    y = forward_return(df, 4)
    cost = pd.Series(1e-4, index=df.index)
    r = walk_forward("ridge", lambda: RidgeForecaster(), f, y, cost, h=4, bars_per_year=6000, n_folds=3)
    assert r.summary["n_oos"] > 0
    assert len(r.folds) == 3
    z = walk_forward("zero", lambda: ZeroForecaster(), f, y, cost, h=4, bars_per_year=6000, n_folds=3)
    assert z.summary["trades"] == 0
    # on pure noise nothing should be significant
    assert abs(r.summary["tstat_nw"]) < 3
