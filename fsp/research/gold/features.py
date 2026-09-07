"""Feature and label construction for gold return forecasting.

Design rules (each has bitten a previous FSP model):
  * Every feature at bar t uses only data with timestamp <= t.
  * Labels are forward LOG returns of the mid close over `h` bars, plus a
    triple-barrier outcome so we can score in the units trading happens in.
  * Macro (daily, NY close) is lagged one full day before joining to intraday
    bars; a bar at 10:00 UTC cannot know that day's NY close.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- indicators

def rsi(s: pd.Series, n: int = 14) -> pd.Series:
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def parkinson_vol(df: pd.DataFrame, n: int) -> pd.Series:
    hl = np.log(df["high"] / df["low"]) ** 2
    return np.sqrt(hl.rolling(n).mean() / (4 * np.log(2)))


def garman_klass_vol(df: pd.DataFrame, n: int) -> pd.Series:
    hl = 0.5 * np.log(df["high"] / df["low"]) ** 2
    co = (2 * np.log(2) - 1) * np.log(df["close"] / df["open"]) ** 2
    return np.sqrt((hl - co).rolling(n).mean())


def adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    up, dn = h.diff(), -l.diff()
    plus = np.where((up > dn) & (up > 0), up, 0.0)
    minus = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / n, adjust=False).mean()
    pdi = 100 * pd.Series(plus, index=df.index).ewm(alpha=1 / n, adjust=False).mean() / atr
    mdi = 100 * pd.Series(minus, index=df.index).ewm(alpha=1 / n, adjust=False).mean() / atr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False).mean()


# ------------------------------------------------------------------ features

def build_features(df: pd.DataFrame, bars_per_day: int) -> pd.DataFrame:
    """Price-derived features. `bars_per_day` scales lookbacks (24 for H1, 1 for D)."""
    c = df["close"]
    r = np.log(c).diff()
    f = pd.DataFrame(index=df.index)

    # momentum / mean reversion at several horizons (in bars)
    for k in (1, 2, 4, 8, 24, 72, 168):
        k_b = max(1, int(round(k * bars_per_day / 24))) if bars_per_day < 24 else k
        f[f"ret_{k}"] = np.log(c / c.shift(k_b))
    # realised vol and its term structure
    d1, d5, d20 = bars_per_day, 5 * bars_per_day, 20 * bars_per_day
    # on daily bars a 1-bar std is undefined; use |r| as the 1-day vol proxy
    f["rv_1d"] = r.rolling(d1).std() if d1 > 1 else r.abs()
    f["rv_5d"] = r.rolling(d5).std()
    f["rv_20d"] = r.rolling(d20).std()
    f["rv_ratio_1_20"] = f["rv_1d"] / f["rv_20d"]
    f["pk_5d"] = parkinson_vol(df, d5)
    f["gk_5d"] = garman_klass_vol(df, d5)
    f["vol_z"] = (f["rv_1d"] - f["rv_20d"]) / f["rv_20d"].rolling(d20 * 3).std()
    # vol-normalised returns (what a model can actually generalise on)
    for k in (1, 4, 24):
        k_b = max(1, int(round(k * bars_per_day / 24))) if bars_per_day < 24 else k
        f[f"zret_{k}"] = f[f"ret_{k}"] / (f["rv_20d"] * np.sqrt(k_b))
    # trend / oscillators
    f["rsi14"] = rsi(c, 14)
    f["rsi_d"] = rsi(c, 14 * bars_per_day) if bars_per_day > 1 else rsi(c, 14)
    f["adx14"] = adx(df, 14)
    for span in (20, 50, 200):
        e = c.ewm(span=span * max(1, bars_per_day // 24) if bars_per_day >= 24 else span, adjust=False).mean()
        f[f"ema{span}_dist"] = np.log(c / e) / f["rv_20d"].replace(0, np.nan)
    # range position
    hi20, lo20 = df["high"].rolling(d20).max(), df["low"].rolling(d20).min()
    f["rng_pos_20d"] = (c - lo20) / (hi20 - lo20).replace(0, np.nan)
    f["bar_range_norm"] = (df["high"] - df["low"]) / c / f["rv_1d"].replace(0, np.nan)
    f["close_loc"] = (c - df["low"]) / (df["high"] - df["low"]).replace(0, np.nan)
    # activity
    v = df["volume"]
    f["vol_rel_5d"] = v / v.rolling(d5).mean()
    if "spread" in df:
        f["spread_bps"] = df["spread"] / c * 1e4
    # calendar (gold has strong intraday seasonality: London/NY/COMEX)
    idx = df.index
    f["hour"] = idx.hour
    f["dow"] = idx.dayofweek
    f["dom"] = idx.day
    f["hour_sin"] = np.sin(2 * np.pi * idx.hour / 24)
    f["hour_cos"] = np.cos(2 * np.pi * idx.hour / 24)
    f["is_london"] = ((idx.hour >= 7) & (idx.hour < 16)).astype(int)
    f["is_ny"] = ((idx.hour >= 13) & (idx.hour < 21)).astype(int)
    f["is_asia"] = ((idx.hour >= 0) & (idx.hour < 7)).astype(int)
    return f


def attach_macro(f: pd.DataFrame, macro: pd.DataFrame, lag_days: int = 1) -> pd.DataFrame:
    """Join daily macro, lagged so each bar only sees closes strictly before its date."""
    m = macro.copy()
    chg = {}
    for col in m.columns:
        chg[f"{col}_chg1"] = np.log(m[col]).diff()
        chg[f"{col}_chg5"] = np.log(m[col]).diff(5)
        chg[f"{col}_z20"] = (m[col] - m[col].rolling(20).mean()) / m[col].rolling(20).std()
    m = pd.concat([m, pd.DataFrame(chg)], axis=1)
    m["gold_silver_ratio"] = np.nan  # filled by caller if gold daily is available
    m = m.shift(lag_days, freq="D")   # value dated D is visible from D+lag_days 00:00 UTC
    m = m.drop(columns=list(macro.columns) + ["gold_silver_ratio"])  # levels are non-stationary
    day = f.index.normalize()
    joined = m.reindex(day, method="ffill")
    joined.index = f.index
    return pd.concat([f, joined], axis=1)


# -------------------------------------------------------------------- labels

def forward_return(df: pd.DataFrame, h: int) -> pd.Series:
    """Log return from close[t] to close[t+h]."""
    c = df["close"]
    return np.log(c.shift(-h) / c)


def triple_barrier(df: pd.DataFrame, h: int, tp_mult: float, sl_mult: float,
                   vol: pd.Series, side: int = 1) -> pd.DataFrame:
    """Outcome of a trade opened at close[t] with barriers at ±k·vol·close.

    Returns columns: label (+1 tp, -1 sl, 0 timeout), ret (realised log return
    incl. barrier fill), bars (holding time). Vectorised per horizon.
    """
    c = df["close"].to_numpy()
    hi = df["high"].to_numpy()
    lo = df["low"].to_numpy()
    n = len(c)
    v = (vol * c).to_numpy()
    label = np.zeros(n, dtype=int)
    ret = np.full(n, np.nan)
    bars = np.full(n, h, dtype=int)
    for i in range(n - h):
        if not np.isfinite(v[i]) or v[i] <= 0:
            continue
        tp = c[i] + side * tp_mult * v[i]
        sl = c[i] - side * sl_mult * v[i]
        seg_hi = hi[i + 1:i + 1 + h]
        seg_lo = lo[i + 1:i + 1 + h]
        if side == 1:
            hit_tp = np.argmax(seg_hi >= tp) if (seg_hi >= tp).any() else h + 1
            hit_sl = np.argmax(seg_lo <= sl) if (seg_lo <= sl).any() else h + 1
        else:
            hit_tp = np.argmax(seg_lo <= tp) if (seg_lo <= tp).any() else h + 1
            hit_sl = np.argmax(seg_hi >= sl) if (seg_hi >= sl).any() else h + 1
        if hit_tp <= h and hit_tp < hit_sl:
            label[i], ret[i], bars[i] = 1, side * np.log(tp / c[i]), hit_tp + 1
        elif hit_sl <= h:
            label[i], ret[i], bars[i] = -1, side * np.log(sl / c[i]), hit_sl + 1
        else:
            ret[i] = side * np.log(c[i + h] / c[i])
    return pd.DataFrame({"label": label, "ret": ret, "bars": bars}, index=df.index)
