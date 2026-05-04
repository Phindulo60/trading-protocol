"""HTML backtest report with equity curve + trade table + summary stats."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from fsp.backtest.engine import BacktestResult

REPORTS = Path.home() / ".fsp" / "reports"


def render(result: BacktestResult, pair: str, label: str = "") -> Path:
    s = result.stats()
    if s.get("total", 0) == 0:
        out = REPORTS / f"backtest_{pair}_{label}_empty.html"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("<h3>No trades taken in this backtest.</h3>")
        return out

    closed = [t for t in result.trades if t.outcome != "open"]
    df = pd.DataFrame([{
        "open_ts": t.open_ts, "close_ts": t.close_ts, "grade": t.grade,
        "dir": t.direction, "session": t.session,
        "entry": t.entry, "sl": t.sl, "tp1": t.tp1, "tp2": t.tp2,
        "outcome": t.outcome, "R": t.weighted_r,
        "cumR": 0, "checks": f"{t.checklist_passed}/{t.checklist_total}",
    } for t in closed])
    df["cumR"] = df["R"].cumsum()

    fig = make_subplots(rows=2, cols=2, specs=[[{"colspan": 2}, None], [{}, {}]],
                        row_heights=[0.55, 0.45],
                        subplot_titles=("Equity Curve (cumulative R)",
                                        "R distribution", "Trades by grade"))
    fig.add_trace(go.Scatter(x=df["close_ts"], y=df["cumR"], mode="lines+markers",
                             name="cumR", line=dict(width=2)), row=1, col=1)
    fig.add_trace(go.Histogram(x=df["R"], nbinsx=30, name="R dist"), row=2, col=1)
    by_grade = df.groupby(["grade", "outcome"]).size().reset_index(name="n")
    for g in ("A+", "A", "B"):
        sub = by_grade[by_grade["grade"] == g]
        if len(sub):
            fig.add_trace(go.Bar(x=sub["outcome"], y=sub["n"], name=g), row=2, col=2)

    summary = (
        f"<b>{pair} {label}</b>"
        f" · trades: {s['total']}"
        f" · wins: {s['wins']} ({s['win_rate']*100:.1f}%)"
        f" · expectancy: {s['expectancy']:+.3f}R/trade"
        f" · PF: {s['profit_factor']:.2f}"
        f" · totalR: {s['total_r']:+.1f}"
        f" · maxDD: {s['max_dd']:.1f}R"
    )
    fig.update_layout(title=summary, template="plotly_dark", height=800, showlegend=True)

    # Trade table HTML
    table_html = df[["open_ts", "close_ts", "grade", "dir", "session",
                     "entry", "sl", "tp1", "tp2", "outcome", "R", "cumR", "checks"]].tail(200).to_html(
        index=False, float_format=lambda x: f"{x:.5f}" if isinstance(x, float) else str(x),
        classes="trades", border=0,
    )

    out = REPORTS / f"backtest_{pair}_{label or 'run'}.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    html = f"""<html><head><meta charset='utf-8'>
<style>
 body {{ background:#0e1117; color:#e6edf3; font-family:ui-monospace,monospace; padding:16px }}
 table.trades {{ border-collapse:collapse; width:100%; font-size:12px }}
 table.trades th, table.trades td {{ padding:4px 8px; border-bottom:1px solid #30363d }}
 table.trades th {{ background:#161b22 }}
</style></head><body>
{fig.to_html(full_html=False, include_plotlyjs='cdn')}
<h3>Last 200 trades</h3>
{table_html}
</body></html>"""
    out.write_text(html)
    return out
