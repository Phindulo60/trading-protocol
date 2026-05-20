"""Walk-forward validation of Session_AsiaHL_fade strategy."""
import time
from dataclasses import asdict
import runner
from runner import M15, H4, run_backtest, compute_stats, simulate_trade
from phase4 import scan_session_fade, run_strategy


def split_data(m15, h4, ratio=0.5):
    n = len(m15)
    split = int(n * ratio)
    split_ts = m15.index[split]
    return (m15.iloc[:split], m15.iloc[split:],
            h4[h4.index < split_ts], h4[h4.index >= split_ts])


def run_strategy_on(scan_fn, scan_kwargs, m15_df, h4_df, max_hold=16, cooldown=4):
    trades = []
    last_signal = -999
    open_until = -1

    for i in range(60, len(m15_df)):
        if i < open_until or i - last_signal < cooldown:
            continue
        ts = m15_df.index[i]
        m15_slice = m15_df.iloc[:i + 1]
        h4_slice = h4_df[h4_df.index <= ts]
        if len(h4_slice) < 12:
            continue
        sig = scan_fn(m15_slice, h4_slice, **scan_kwargs)
        if sig is None:
            continue
        future = m15_df.iloc[i + 1:]
        t = simulate_trade(sig, future, max_hold)
        trades.append(t)
        last_signal = i
        open_until = i + t.bars_held + 1
    return trades


def main():
    t0 = time.time()
    train_m15, test_m15, train_h4, test_h4 = split_data(M15, H4, 0.5)

    print(f"Walk-forward validation of Session_AsiaHL_fade")
    print(f"Train: {train_m15.index[0]} to {train_m15.index[-1]}")
    print(f"Test:  {test_m15.index[0]} to {test_m15.index[-1]}\n")

    train_trades = run_strategy_on(scan_session_fade, {}, train_m15, train_h4)
    test_trades = run_strategy_on(scan_session_fade, {}, test_m15, test_h4)

    train_s = compute_stats(train_trades)
    test_s = compute_stats(test_trades)

    print(f"TRAIN: n={train_s['n']:>3} WR={train_s['wr']*100:5.1f}% "
          f"Exp={train_s['exp']:+.2f}R PF={train_s['pf']:5.2f} "
          f"R={train_s['total_r']:+6.1f} DD={train_s['max_dd']:+5.1f}")
    print(f"TEST:  n={test_s['n']:>3} WR={test_s['wr']*100:5.1f}% "
          f"Exp={test_s['exp']:+.2f}R PF={test_s['pf']:5.2f} "
          f"R={test_s['total_r']:+6.1f} DD={test_s['max_dd']:+5.1f}")

    flag = "✅ ROBUST" if (train_s['total_r'] > 0 and test_s['total_r'] > 0
                          and train_s['exp'] > 0 and test_s['exp'] > 0) else "⚠️ FRAGILE"
    print(f"\n{flag}")
    print(f"Elapsed: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
