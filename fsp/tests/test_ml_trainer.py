"""Tests for fsp.ml.trainer — GBM meta-model training pipeline.

Uses small synthetic datasets so tests run in <2s and dont need real data.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Skip whole module if lightgbm not installed (CI without [ml] extras)
pytest.importorskip("lightgbm")
pytest.importorskip("sklearn")

from fsp.ml.trainer import (  # noqa: E402
    CalibratedModel,
    TrainingResult,
    _calibration_table,
    _expected_calibration_error,
    _high_p_calibration_gap,
    _prepare_xy,
    _walk_forward_cv,
    load_model,
    train_meta_model,
)
from fsp.ml.features import CATEGORICAL_FEATURES, FEATURE_NAMES  # noqa: E402


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def synthetic_data():
    """Synthetic training set: enough signal that AUC > 0.6, ~64% win rate.

    Builds all 28 features from FEATURE_NAMES. h4_trend_score is the only
    feature that actually drives the label, so the model should easily
    detect signal.
    """
    rng = np.random.default_rng(42)
    n = 400

    df = pd.DataFrame({
        "ts": pd.date_range("2025-01-01", periods=n, freq="2h", tz="UTC"),
    })

    # Categorical features
    df["strategy"] = rng.choice(["TREND_RSI", "ASIA_HL"], size=n)
    df["direction"] = rng.choice(["LONG", "SHORT"], size=n)
    df["pair"] = rng.choice(["USDCAD", "EURUSD", "GBPUSD"], size=n)
    df["session"] = rng.choice(["LONDON", "NY_AM", "ASIA"], size=n)
    df["dow"] = rng.choice(["Mon", "Tue", "Wed", "Thu", "Fri"], size=n)

    # Technical
    h4 = rng.normal(0, 1, size=n)
    df["rsi_at_entry"] = rng.uniform(20, 80, size=n)
    df["rsi_depth"] = rng.uniform(-30, 30, size=n)
    df["atr_pctl_60d"] = rng.uniform(0, 1, size=n)
    df["adx_h4"] = rng.uniform(15, 40, size=n)
    df["h4_trend_score"] = h4
    df["h1_trend_score"] = rng.normal(0, 1, size=n)
    df["m15_atr_norm"] = rng.uniform(0.5, 1.5, size=n)
    df["adr_pct_used"] = rng.uniform(0.0, 0.8, size=n)
    df["bars_since_extreme"] = rng.integers(0, 50, size=n)
    df["range_compress"] = rng.uniform(0.5, 1.5, size=n)

    # Strategy state
    df["strat_wr_last_20"] = rng.uniform(0.4, 0.8, size=n)
    df["strat_streak_signed"] = rng.integers(-5, 5, size=n)
    df["strat_net_r_last_20"] = rng.normal(2.0, 5.0, size=n)
    df["pair_strat_wr_last_10"] = rng.uniform(0.3, 0.9, size=n)
    df["days_since_last_signal"] = rng.uniform(0, 14, size=n)

    # News + macro
    df["mins_to_next_high_event"] = rng.uniform(0, 600, size=n)
    df["high_events_in_24h"] = rng.integers(0, 5, size=n)
    df["dxy_change_24h_pct"] = rng.normal(0, 0.5, size=n)
    df["dxy_trend_aligned"] = rng.choice([0, 1], size=n)

    # LLM
    df["llm_confidence"] = rng.uniform(0.5, 1.0, size=n)
    df["llm_decision_take"] = rng.choice([0, 1], size=n)
    df["llm_decision_skip"] = rng.choice([0, 1], size=n)
    df["llm_decision_reduce"] = rng.choice([0, 1], size=n)

    # Label: P(win) = 0.65 + 0.10 * h4_trend_score, clipped
    p_win = np.clip(0.65 + 0.10 * h4, 0.20, 0.95)
    df["won"] = rng.binomial(1, p_win)
    df["won_tp"] = df["won"]
    df["r_outcome"] = np.where(df["won"] == 1, 2.0, -1.0)

    return df


# ── Helper functions ────────────────────────────────────────────────────────

def test_prepare_xy_keeps_categorical(synthetic_data):
    X, y = _prepare_xy(synthetic_data, label_col="won")
    assert list(X.columns) == FEATURE_NAMES
    for c in CATEGORICAL_FEATURES:
        assert str(X[c].dtype) == "category"
    assert y.dtype == int
    assert set(y.unique()) <= {0, 1}


def test_calibration_table_shape():
    rng = np.random.default_rng(1)
    n = 100
    p = rng.uniform(0, 1, size=n)
    y = rng.binomial(1, p)
    tbl = _calibration_table(y, p, n_bins=5)
    assert {"bin", "n", "avg_pred", "actual_wr", "calibration_gap_pp"} <= set(tbl.columns)
    assert len(tbl) <= 5
    # n must sum to total
    assert tbl["n"].sum() == n


def test_ece_is_weighted_avg():
    """ECE = Σ(|gap| × n) / Σn."""
    tbl = pd.DataFrame({
        "n": [50, 50],
        "calibration_gap_pp": [10.0, -20.0],  # |gaps| = 10, 20
    })
    ece = _expected_calibration_error(tbl)
    # weighted: (10*50 + 20*50) / 100 = 15
    assert abs(ece - 15.0) < 1e-9


def test_high_p_gap_filters_above_threshold():
    """High-P gap only counts bins where avg_pred ≥ threshold."""
    tbl = pd.DataFrame({
        "n": [10, 20, 30],
        "avg_pred": [0.40, 0.60, 0.80],
        "calibration_gap_pp": [50.0, 10.0, 5.0],  # only last 2 above 0.55
    })
    gap = _high_p_calibration_gap(tbl, threshold=0.55)
    # weighted: (10*20 + 5*30) / 50 = (200 + 150) / 50 = 7
    assert abs(gap - 7.0) < 1e-9


def test_high_p_gap_zero_when_no_high_bins():
    """No bins above threshold returns 0.0 (cant fail acceptance)."""
    tbl = pd.DataFrame({
        "n": [10, 20],
        "avg_pred": [0.40, 0.50],
        "calibration_gap_pp": [99.0, 99.0],
    })
    gap = _high_p_calibration_gap(tbl, threshold=0.65)
    assert gap == 0.0


# ── Walk-forward CV ─────────────────────────────────────────────────────────

def test_walk_forward_cv_returns_n_folds(synthetic_data):
    cv_aucs, holdout_df, train_df = _walk_forward_cv(
        synthetic_data, label_col="won", n_splits=3,
        lgbm_params={"n_estimators": 30, "verbose": -1, "random_state": 42},
    )
    assert len(cv_aucs) == 3
    # All folds should produce valid AUCs
    for auc in cv_aucs:
        assert 0.0 <= auc <= 1.0


def test_walk_forward_cv_holdout_separated(synthetic_data):
    """Holdout must be the most recent 20%, train_df is the prior 80%."""
    _, holdout_df, train_df = _walk_forward_cv(
        synthetic_data, label_col="won", n_splits=3,
        lgbm_params={"n_estimators": 30, "verbose": -1, "random_state": 42},
    )
    # No overlap, holdout is more recent
    assert holdout_df["ts"].min() >= train_df["ts"].max()
    # Sizes correct
    assert len(train_df) + len(holdout_df) == len(synthetic_data)
    assert len(holdout_df) == int(len(synthetic_data) * 0.20)


# ── End-to-end training ─────────────────────────────────────────────────────

def test_train_meta_model_end_to_end(synthetic_data, tmp_path):
    """Full pipeline: parquet → CV → calibration → save → load."""
    parquet = tmp_path / "train.parquet"
    synthetic_data.to_parquet(parquet)
    output = tmp_path / "model.pkl"

    result = train_meta_model(
        parquet_path=parquet,
        label_col="won",
        output_path=output,
        n_cv_splits=3,
        lgbm_params={"n_estimators": 50, "verbose": -1, "random_state": 42},
    )

    assert isinstance(result, TrainingResult)
    assert isinstance(result.model, CalibratedModel)
    assert len(result.cv_aucs) == 3
    assert 0.4 < result.mean_cv_auc < 1.0  # signal exists
    assert 0.4 < result.holdout_auc < 1.0
    assert result.n_train + result.n_holdout == len(synthetic_data)
    assert output.exists()


def test_saved_model_round_trip(synthetic_data, tmp_path):
    """Save → load → predict produces same probabilities."""
    parquet = tmp_path / "train.parquet"
    synthetic_data.to_parquet(parquet)
    output = tmp_path / "model.pkl"

    result = train_meta_model(
        parquet_path=parquet, label_col="won", output_path=output,
        n_cv_splits=3,
        lgbm_params={"n_estimators": 30, "verbose": -1, "random_state": 42},
    )

    # Load via load_model() helper
    bundle = load_model(output)
    assert "model" in bundle
    assert "feature_names" in bundle
    assert bundle["feature_names"] == FEATURE_NAMES

    # Predictions should match (within fp tolerance)
    X, _ = _prepare_xy(synthetic_data.head(20), label_col="won")
    p_orig = result.model.predict_proba(X)[:, 1]
    p_loaded = bundle["model"].predict_proba(X)[:, 1]
    np.testing.assert_allclose(p_orig, p_loaded, rtol=1e-9)


def test_calibrated_model_outputs_valid_probabilities(synthetic_data, tmp_path):
    """predict_proba must return valid 2-column probability matrix in [0,1]."""
    parquet = tmp_path / "train.parquet"
    synthetic_data.to_parquet(parquet)

    result = train_meta_model(
        parquet_path=parquet, label_col="won",
        output_path=tmp_path / "m.pkl", n_cv_splits=3,
        lgbm_params={"n_estimators": 30, "verbose": -1, "random_state": 42},
    )

    X, _ = _prepare_xy(synthetic_data, label_col="won")
    proba = result.model.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert (proba >= 0).all() and (proba <= 1).all()
    # Each row sums to 1
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-6)


def test_load_model_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_model(tmp_path / "does_not_exist.pkl")
