"""Render an interactive Plotly HTML chart with everything detected."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

from fsp.context.levels import htf_levels, mark_swept, monday_range
from fsp.context.sessions import session_ranges, annotate_sessions
from fsp.data.feed import default_feed
from fsp.structure.displacement import find_displacements
from fsp.structure.fvg import find_fvgs, mark_mitigation
from fsp.structure.order_blocks import find_order_blocks, mark_ob_mitigation
from fsp.structure.swings import find_swings, mark_broken


REPORTS = Path.home() / ".fsp" / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)


def render(pair: str, tf: str, days: int, feed_kind: str = "duka",
           swing_len: int = 5, open_browser: bool = True) -> Path:
    feed = default_feed(feed_kind)
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    df = feed.history(pair, tf, start, end)  # type: ignore[arg-type]
    if df.empty:
        raise SystemExit(f"No data for {pair} {tf}")

    swings = mark_broken(find_swings(df, length=swing_len), df)
    fvgs = mark_mitigation(find_fvgs(df, tf=tf), df)  # type: ignore[arg-type]
    obs = mark_ob_mitigation(find_order_blocks(df, tf=tf), df)  # type: ignore[arg-type]
    disps = find_displacements(df)
    levels = mark_swept(htf_levels(df), df)
    mr = monday_range(df) or {}

    fig = go.Figure(data=[go.Candlestick(
        x=df.index, open=df["open"], high=df["high"], low=df["low"], close=df["close"],
        name=pair, increasing_line_color="#26a69a", decreasing_line_color="#ef5350")])

    x0, x1 = df.index[0], df.index[-1]

    # FVGs as shaded rectangles (only unmitigated drawn strong, rest faded)
    for f in fvgs:
        color = "rgba(38,166,154,0.25)" if f.direction == "bull" else "rgba(239,83,80,0.25)"
        if f.mitigated:
            color = color.replace("0.25", "0.07")
        fig.add_shape(type="rect", x0=f.ts, x1=x1, y0=f.bottom, y1=f.top,
                      line=dict(width=0), fillcolor=color, layer="below")

    # Order blocks
    for ob in obs:
        color = "rgba(76,175,80,0.30)" if ob.direction == "bull" else "rgba(244,67,54,0.30)"
        if ob.mitigated:
            color = color.replace("0.30", "0.10")
        fig.add_shape(type="rect", x0=ob.ts, x1=x1, y0=ob.bottom, y1=ob.top,
                      line=dict(color="orange", width=1, dash="dot"),
                      fillcolor=color, layer="below")

    # Swings as markers
    for s in swings:
        fig.add_trace(go.Scatter(
            x=[s.ts], y=[s.price],
            mode="markers+text",
            marker=dict(size=8, color="red" if s.kind == "high" else "green",
                        symbol="triangle-down" if s.kind == "high" else "triangle-up"),
            text=[("SH" if s.strong else "wh") if s.kind == "high" else ("SL" if s.strong else "wl")],
            textposition="top center" if s.kind == "high" else "bottom center",
            textfont=dict(size=9),
            showlegend=False, hovertext=f"{s.kind} @ {s.price:.5f} strong={s.strong} broken={s.broken}",
        ))

    # Displacement markers
    for d in disps:
        fig.add_trace(go.Scatter(
            x=[d.ts], y=[d.close],
            mode="markers",
            marker=dict(size=10, color="lime" if d.direction == "up" else "magenta",
                        symbol="diamond"),
            name=f"disp-{d.direction}",
            showlegend=False,
            hovertext=f"Displacement {d.direction} {d.atr_mult:.1f}× ATR",
        ))

    # HTF levels as horizontal lines
    level_colors = {
        "PDH": "red", "PDL": "green", "PWH": "darkred", "PWL": "darkgreen",
        "PMH": "maroon", "PML": "teal", "DO": "gold", "WO": "orange",
    }
    all_levels = {**levels, **mr}
    for name, lvl in all_levels.items():
        fig.add_hline(
            y=lvl.price, line=dict(color=level_colors.get(name, "gray"),
                                   dash="dash" if "W" in name or "M" in name else "solid",
                                   width=1),
            annotation_text=f"{name} {'(swept)' if lvl.swept else ''}",
            annotation_position="right",
        )

    # Session shading (last ~5 days only to avoid clutter)
    annotated = annotate_sessions(df)
    recent_start = df.index[-1] - pd.Timedelta(days=5)
    recent = annotated[annotated.index >= recent_start]
    sess_colors = {
        "ASIA": "rgba(0,200,255,0.08)", "LO": "rgba(33,150,243,0.10)",
        "NY-AM": "rgba(255,152,0,0.10)", "LUNCH": "rgba(158,158,158,0.08)",
        "NY-PM": "rgba(121,85,72,0.08)",
    }
    cur_sess = None
    seg_start = None
    for ts, sess in zip(recent.index, recent["session"]):
        if sess != cur_sess:
            if cur_sess and cur_sess in sess_colors:
                fig.add_vrect(x0=seg_start, x1=ts, fillcolor=sess_colors[cur_sess],
                              line_width=0, layer="below")
            cur_sess, seg_start = sess, ts
    if cur_sess and cur_sess in sess_colors:
        fig.add_vrect(x0=seg_start, x1=recent.index[-1], fillcolor=sess_colors[cur_sess],
                      line_width=0, layer="below")

    fig.update_layout(
        title=f"{pair} {tf} — {days}d  |  swings:{len(swings)}  FVGs:{len(fvgs)} (unmit:{sum(1 for f in fvgs if not f.mitigated)})  OBs:{len(obs)}  disp:{len(disps)}",
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        height=800,
        hovermode="x unified",
    )

    out = REPORTS / f"{pair}_{tf}_{days}d.html"
    fig.write_html(out, include_plotlyjs="cdn")
    if open_browser:
        import webbrowser
        webbrowser.open(f"file://{out}")
    return out
