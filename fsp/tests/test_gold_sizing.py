import numpy as np, pandas as pd
from fsp.research.gold.sizing import size_lots, simulate_path, monte_carlo, SizingConfig


def test_size_lots_grid_and_caps():
    cfg = SizingConfig(start_equity=1000, risk_pct=0.01, max_lot=0.10)
    # $10 risk budget, $5/oz stop -> 2 oz = 0.02 lot
    assert size_lots(1000, 5.0, cfg, 4400) == 0.02
    # huge equity -> capped at 0.10
    assert size_lots(1_000_000, 5.0, cfg, 4400) == 0.10
    # tiny equity, min lot would risk > 3x target -> skip
    assert size_lots(200, 20.0, cfg, 4400) == 0.0
    # min lot risks <= 3x target -> trade the minimum
    assert size_lots(500, 10.0, cfg, 4400) == 0.01
    assert size_lots(1000, 0.0, cfg, 4400) == 0.0


def test_margin_limits_lots():
    cfg = SizingConfig(start_equity=1000, risk_pct=0.05, max_lot=0.10, margin_pct=0.05)
    # 0.10 lot = 10 oz * 4400 * 5% = $2200 margin > 80% of $1000 -> must shrink
    lots = size_lots(1000, 5.0, cfg, 4400)
    assert lots * 100 * 4400 * 0.05 <= 800 + 1e-6


def test_path_compounds_and_throttles():
    cfg = SizingConfig(start_equity=1000, risk_pct=0.02, dd_throttle=0.05)
    r = np.array([-1, -1, -1, -1, 2, 2, 2])
    p = simulate_path(r, np.full(7, 5.0), np.full(7, 4400.0), cfg)
    assert p.throttled.iloc[4]  # after ~8% drawdown risk is halved
    assert p.equity.iloc[-1] > p.equity.iloc[3]
    assert (p.lots <= 0.10).all()


def test_monte_carlo_shape():
    tr = pd.DataFrame({"r_multiple": np.r_[np.full(30, 2.0), np.full(30, -1.0)],
                       "fill": 4400.0, "sl": 4395.0})
    cfg = SizingConfig()
    mc = monte_carlo(tr, cfg, n_paths=200, horizon=60)
    assert 0 <= mc["P(dead)"] <= 1 and mc["median_final"] > 0
