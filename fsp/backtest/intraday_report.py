"""HTML report for intraday strategies backtest results."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from fsp.backtest.engine import BacktestResult

REPORTS = Path.home() / ".fsp" / "reports"
STRATEGY_COLORS = {"ECM": "#36a2eb", "ARB": "#ff9f40"}


def render_intraday(results: dict[str, BacktestResult],
                    pair: str, label: str = "") -> Path:
    """Render a single HTML page comparing ECM and ARB results side by side."""
    REPORTS.mkdir(parents=True, exist_ok=True)
    out = REPORTS / f"intraday_{pair}_{label or 'run'}.html"

    # Build per-strategy dataframes
    dfs: dict[str, pd.DataFrame] = {}
    stats: dict[str, dict] = {}
    for strat, res in results.items():
        st = res.stats()
        stats[strat] = st
        closed = [t for t in res.trades if t.outcome not in ("open", "eop", "pending")]
        if not closed:
            dfs[strat] = pd.DataFrame()
            continue
        df = pd.DataFrame([{
            "ts": t.close_ts or t.open_ts,
            "open_ts": t.open_ts,
            "dir": t.direction,
            "session": t.session,
            "dow": ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][t.dow] if t.dow < 7 else "?",
            "entry": t.entry,
            "sl": t.sl,
            "tp1": t.tp1,
            "outcome": t.outcome,
            "R": t.weighted_r,
            "bars": t.bars_held,
        } for t in closed])
        df["cumR"] = df["R"].cumsum()
        dfs[strat] = df

    n_strats = len([s for s in results if len(dfs.get(s, [])) > 0])
    if n_strats == 0:
        out.write_text("<h3>No trades taken in this backtest.</h3>")
        return out

    # ── Figure: equity curves side by side + R distributions ──────────────────
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Equity Curve — ECM", "Equity Curve — ARB",
            "R Distribution — ECM", "R Distribution — ARB",
        ),
        row_heights=[0.55, 0.45],
    )

    for col_idx, (strat, color) in enumerate(STRATEGY_COLORS.items(), start=1):
        df = dfs.get(strat)
        if df is None or len(df) == 0:
            continue
        fig.add_trace(go.Scatter(x=df["ts"], y=df["cumR"], mode="lines+markers",
                                  name=f"{strat} cumR", line=dict(color=color, width=2)),
                      row=1, col=col_idx)
        fig.add_trace(go.Histogram(x=df["R"], nbinsx=20, name=f"{strat} R dist",
                                    marker_color=color),
                      row=2, col=col_idx)

    fig.update_layout(template="plotly_dark", height=750, showlegend=True)

    # ── Summary HTML ───────────────────────────────────────────────────────────
    summary_rows = []
    for strat in ("ECM", "ARB"):
        st = stats.get(strat, {})
        n = st.get("total", 0)
        if n == 0:
            summary_rows.append(f"<tr><td>{strat}</td><td colspan='7'>No trades</td></tr>")
            continue
        by_sess = st.get("by_session", {})
        best_sess = max(by_sess, key=lambda k: by_sess[k].get("exp", -99)) if by_sess else "-"
        summary_rows.append(
            f"<tr>"
            f"<td><b>{strat}</b></td>"
            f"<td>{n}</td>"
            f"<td>{st['win_rate']*100:.1f}%</td>"
            f"<td>{st['expectancy']:+.3f}R</td>"
            f"<td>{st['profit_factor']:.2f}</td>"
            f"<td>{st['total_r']:+.1f}R</td>"
            f"<td>{st['max_dd']:.1f}R</td>"
            f"<td>{best_sess}</td>"
            f"</tr>"
        )

    # Per-session breakdown table
    session_html_parts = []
    for strat in ("ECM", "ARB"):
        st = stats.get(strat, {})
        by_sess = st.get("by_session", {})
        if not by_sess:
            continue
        rows_html = "".join(
            f"<tr><td>{s}</td><td>{d['n']}</td>"
            f"<td>{d['wr']*100:.0f}%</td><td>{d['exp']:+.3f}R</td></tr>"
            for s, d in sorted(by_sess.items(), key=lambda x: -x[1].get("exp", 0))
        )
        session_html_parts.append(f"""
        <h4>{strat} — by session</h4>
        <table class='trades'>
          <tr><th>session</th><th>n</th><th>wr%</th><th>exp</th></tr>
          {rows_html}
        </table>""")

    # Trade tables (last 100 per strategy)
    trade_tables = []
    for strat in ("ECM", "ARB"):
        df = dfs.get(strat)
        if df is None or len(df) == 0:
            continue
        tbl = df[["open_ts","dir","session","dow","entry","sl","tp1","outcome","R","cumR","bars"]].tail(100)
        trade_tables.append(f"<h3>{strat} — Last {min(100, len(df))} trades</h3>" +
                             tbl.to_html(index=False, classes="trades", border=0,
                                          float_format=lambda x: f"{x:.5f}" if abs(x) > 0.001 else f"{x:.3f}"))

    summary_table = f"""
    <table class='trades'>
      <tr><th>Strategy</th><th>Trades</th><th>WR%</th><th>Exp/trade</th>
          <th>PF</th><th>Total R</th><th>Max DD</th><th>Best session</th></tr>
      {''.join(summary_rows)}
    </table>"""

    html = f"""<html><head><meta charset='utf-8'>
<style>
 body {{background:#0e1117;color:#e6edf3;font-family:ui-monospace,monospace;padding:16px}}
 table.trades {{border-collapse:collapse;width:100%;font-size:12px;margin:12px 0}}
 table.trades th,table.trades td {{padding:4px 8px;border-bottom:1px solid #30363d}}
 table.trades th {{background:#161b22}}
 h3 {{color:#58a6ff;margin-top:24px}} h4 {{color:#79c0ff}}
</style></head><body>
<h2>Intraday Backtest — {pair} — {label}</h2>
{summary_table}
{fig.to_html(full_html=False, include_plotlyjs='cdn')}
<div style="display:flex;gap:24px">{''.join(session_html_parts)}</div>
{''.join(trade_tables)}
</body></html>"""

    out.write_text(html)
    return out
