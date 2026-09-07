"""Walk-forward evaluation for gold forecasters.

A forecaster is anything with fit(X, y) / predict(X). Evaluation is
expanding-window walk-forward with a purge gap of `h` bars between train
and test so overlapping forward-return labels cannot leak.

Metrics per fold and pooled:
  * directional accuracy vs the 50% / majority-class baseline
  * information coefficient (Spearman of prediction vs realised return)
  * MSE skill vs zero forecast (skill = 1 - MSE_model / MSE_zero)
  * trading P&L: take sign(pred) when |pred| > threshold, hold h bars,
    pay observed spread once per round trip, report mean R, Sharpe and
    t-stat over trades

Every number is out-of-sample. If pooled t-stat < 2 there is no edge.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


class Forecaster(Protocol):
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None: ...
    def predict(self, X: pd.DataFrame) -> np.ndarray: ...


@dataclass
class FoldResult:
    fold: int
    n: int
    dir_acc: float
    ic: float
    mse_skill: float
    trades: int
    mean_ret_bps: float
    sharpe_ann: float
    tstat: float


@dataclass
class EvalResult:
    name: str
    horizon: int
    folds: list[FoldResult]
    pred: pd.Series
    y: pd.Series
    trade_ret: pd.Series  # net log return per taken trade (0 when flat)
    summary: dict = field(default_factory=dict)

    def table(self) -> pd.DataFrame:
        return pd.DataFrame([f.__dict__ for f in self.folds])


def nw_tstat(x: np.ndarray, lag: int) -> float:
    """Newey-West t-stat of the mean of an autocorrelated series.

    Positions are held h bars but sampled every bar, so consecutive returns
    overlap and a naive t-stat is inflated by ~sqrt(h). Bartlett kernel, lag=h-1.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 10:
        return np.nan
    mu = x.mean()
    e = x - mu
    s = e @ e / n
    for L in range(1, min(lag, n - 1) + 1):
        w = 1 - L / (lag + 1)
        s += 2 * w * (e[L:] @ e[:-L]) / n
    if s <= 0:
        return np.nan
    return float(mu / np.sqrt(s / n))


def _trade_stats(pred: np.ndarray, y: np.ndarray, cost: np.ndarray, thr: float,
                 h: int, bars_per_year: float) -> tuple[int, float, float, float, np.ndarray]:
    take = np.abs(pred) > thr
    side = np.sign(pred)
    net = np.where(take, side * y - cost, 0.0)
    tr = net[take]
    if len(tr) < 2:
        return int(take.sum()), np.nan, np.nan, np.nan, net
    mean, sd = tr.mean(), tr.std(ddof=1)
    # t-stat on the full per-bar series (zeros when flat) with h-1 overlap correction
    t = nw_tstat(net, h - 1)
    # Sharpe from non-overlapping h-bar blocks of the per-bar series
    blocks = net[:len(net) // h * h].reshape(-1, h).sum(axis=1)
    sharpe = blocks.mean() / blocks.std(ddof=1) * np.sqrt(bars_per_year / h) if blocks.std(ddof=1) > 0 else np.nan
    return len(tr), mean * 1e4, sharpe, t, net


def walk_forward(
    name: str,
    make_model: Callable[[], Forecaster],
    X: pd.DataFrame,
    y: pd.Series,
    cost: pd.Series,
    h: int,
    bars_per_year: float,
    n_folds: int = 6,
    min_train_frac: float = 0.4,
    threshold: float | Callable[[np.ndarray], float] = 0.0,
) -> EvalResult:
    """Expanding-window walk-forward with an h-bar purge gap.

    `threshold` may be a float in units of y, or a callable applied to the
    TRAIN-set predictions (e.g. lambda p: np.quantile(np.abs(p), 0.5)) so the
    trade filter never peeks at test data.
    """
    mask = X.notna().all(axis=1) & y.notna()
    X, y, cost = X[mask], y[mask], cost[mask]
    n = len(y)
    first_test = int(n * min_train_frac)
    edges = np.linspace(first_test, n, n_folds + 1).astype(int)

    folds: list[FoldResult] = []
    preds = pd.Series(np.nan, index=y.index)
    net_all = pd.Series(0.0, index=y.index)
    for k in range(n_folds):
        te0, te1 = edges[k], edges[k + 1]
        tr1 = te0 - h  # purge: last h train rows have labels overlapping the test window
        Xtr, ytr = X.iloc[:tr1], y.iloc[:tr1]
        Xte, yte, cte = X.iloc[te0:te1], y.iloc[te0:te1], cost.iloc[te0:te1]
        m = make_model()
        m.fit(Xtr, ytr)
        p = np.asarray(m.predict(Xte), dtype=float)
        thr = threshold(np.asarray(m.predict(Xtr), dtype=float)) if callable(threshold) else threshold
        yv = yte.to_numpy()
        dir_acc = float((np.sign(p) == np.sign(yv)).mean())
        ic = float(spearmanr(p, yv).correlation) if np.std(p) > 0 else 0.0
        mse_skill = float(1 - np.mean((yv - p) ** 2) / np.mean(yv ** 2))
        ntr, mean_bps, sharpe, t, net = _trade_stats(p, yv, cte.to_numpy(), thr, h, bars_per_year)
        folds.append(FoldResult(k, len(yv), dir_acc, ic, mse_skill, ntr, mean_bps, sharpe, t))
        preds.iloc[te0:te1] = p
        net_all.iloc[te0:te1] = net

    valid = preds.notna()
    p_all, y_all = preds[valid].to_numpy(), y[valid].to_numpy()
    taken = net_all[valid] != 0
    tr = net_all[valid][taken]
    net_v = net_all[valid].to_numpy()
    pooled_t = nw_tstat(net_v, h - 1)
    # excess over always-long: the only baseline that matters in a 3.5x bull run
    long_net = y_all - cost[valid].to_numpy()
    excess = net_v - long_net
    blocks = net_v[:len(net_v) // h * h].reshape(-1, h).sum(axis=1)
    sharpe = blocks.mean() / blocks.std(ddof=1) * np.sqrt(bars_per_year / h) if len(blocks) > 2 and blocks.std(ddof=1) > 0 else np.nan
    summary = {
        "name": name, "h": h, "n_oos": int(valid.sum()),
        "dir_acc": float((np.sign(p_all) == np.sign(y_all)).mean()),
        "base_rate_up": float((y_all > 0).mean()),
        "ic": float(spearmanr(p_all, y_all).correlation) if np.std(p_all) > 0 else 0.0,
        "mse_skill": float(1 - np.mean((y_all - p_all) ** 2) / np.mean(y_all ** 2)),
        "trades": int(taken.sum()),
        "mean_ret_bps": float(tr.mean() * 1e4) if len(tr) else np.nan,
        "total_ret_pct": float(tr.sum() * 100) if len(tr) else 0.0,
        "tstat_nw": float(pooled_t),
        "sharpe": float(sharpe),
        "excess_vs_long_bps": float(excess.mean() * 1e4),
        "excess_t_nw": nw_tstat(excess, h - 1),
        "win_rate": float((tr > 0).mean()) if len(tr) else np.nan,
    }
    return EvalResult(name, h, folds, preds, y, net_all, summary)


# ------------------------------------------------------------------ baselines

class ZeroForecaster:
    def fit(self, X, y): pass
    def predict(self, X): return np.zeros(len(X))


class MeanForecaster:
    """Unconditional drift: the honest 'gold goes up' baseline."""
    def fit(self, X, y): self.mu = float(y.mean())
    def predict(self, X): return np.full(len(X), self.mu)


class MomentumForecaster:
    """Sign of trailing k-bar return scaled to label vol; the classic TSMOM rule."""
    def __init__(self, col: str): self.col = col
    def fit(self, X, y): self.scale = float(y.std())
    def predict(self, X): return np.sign(X[self.col].to_numpy()) * self.scale * 0.1


class RidgeForecaster:
    def __init__(self, alpha: float = 10.0):
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        self.m = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    def fit(self, X, y): self.m.fit(X, y)
    def predict(self, X): return self.m.predict(X)


class LGBMForecaster:
    def __init__(self, **params):
        from lightgbm import LGBMRegressor
        base = dict(n_estimators=300, learning_rate=0.02, num_leaves=15, max_depth=4,
                    min_child_samples=100, subsample=0.8, subsample_freq=1,
                    colsample_bytree=0.7, reg_lambda=5.0, verbose=-1)
        base.update(params)
        self.m = LGBMRegressor(**base)
    def fit(self, X, y): self.m.fit(X, y)
    def predict(self, X): return self.m.predict(X)
    @property
    def importances(self): return self.m.feature_importances_


class LGBMClassifierForecaster:
    """Predicts P(up) and emits (p - 0.5) so sign/threshold logic still applies."""
    def __init__(self, **params):
        from lightgbm import LGBMClassifier
        base = dict(n_estimators=300, learning_rate=0.02, num_leaves=15, max_depth=4,
                    min_child_samples=100, subsample=0.8, subsample_freq=1,
                    colsample_bytree=0.7, reg_lambda=5.0, verbose=-1)
        base.update(params)
        self.m = LGBMClassifier(**base)
    def fit(self, X, y): self.m.fit(X, (y > 0).astype(int))
    def predict(self, X): return self.m.predict_proba(X)[:, 1] - 0.5
