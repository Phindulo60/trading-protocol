"""fsp CLI entrypoint."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

import typer
from rich import print
from rich.table import Table

from fsp.context.levels import htf_levels, mark_swept, monday_range
from fsp.context.sessions import session_ranges
from fsp.data.feed import default_feed
from fsp.data.types import Grade
from fsp.structure.displacement import find_displacements
from fsp.structure.fvg import find_fvgs, mark_mitigation
from fsp.structure.order_blocks import find_order_blocks, mark_ob_mitigation
from fsp.structure.swings import find_swings, mark_broken

app = typer.Typer(add_completion=False, help="4-Step Protocol engine (fsp)")


@app.command()
def fetch(
    pair: str = typer.Option("EURUSD", help="EURUSD | GBPUSD"),
    tf: str = typer.Option("H1", help="M5 M15 M30 H1 H4 D"),
    days: int = typer.Option(90, help="Days of history to pull"),
    feed: str = typer.Option("duka", help="duka | yf"),
):
    """Download bars and print the tail."""
    f = default_feed(feed)
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    df = f.history(pair, tf, start, end)  # type: ignore[arg-type]
    print(f"[bold green]Fetched[/] {len(df)} bars of {pair} {tf} from {feed}")
    print(df.tail(10))


@app.command()
def scan(
    pair: str = typer.Option("EURUSD"),
    tf: str = typer.Option("H1"),
    days: int = typer.Option(30),
    feed: str = typer.Option("duka"),
    swing_len: int = typer.Option(5),
):
    """Detect swings + FVGs on the given pair/TF and print a summary."""
    f = default_feed(feed)
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    df = f.history(pair, tf, start, end)  # type: ignore[arg-type]
    swings = mark_broken(find_swings(df, length=swing_len), df)
    fvgs = mark_mitigation(find_fvgs(df, tf=tf), df)  # type: ignore[arg-type]

    t = Table(title=f"{pair} {tf} — last 20 swings")
    for col in ("ts", "kind", "price", "strong", "broken"):
        t.add_column(col)
    for s in swings[-20:]:
        t.add_row(s.ts.strftime("%Y-%m-%d %H:%M"), s.kind, f"{s.price:.5f}",
                  "★" if s.strong else "", "✔" if s.broken else "")
    print(t)

    active = [f for f in fvgs if not f.mitigated]
    print(f"\n[bold]FVGs[/]: {len(fvgs)} total, {len(active)} unmitigated")
    for f_ in active[-10:]:
        print(f"  {f_.ts:%Y-%m-%d %H:%M}  {f_.direction:<4} "
              f"[{f_.bottom:.5f} – {f_.top:.5f}]  inv={f_.inverted}")


if __name__ == "__main__":
    app()


@app.command()
def levels(
    pair: str = typer.Option("EURUSD"),
    tf: str = typer.Option("H1"),
    days: int = typer.Option(90),
    feed: str = typer.Option("duka"),
):
    """Print current HTF liquidity levels + Monday range."""
    f = default_feed(feed)
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    df = f.history(pair, tf, start, end)  # type: ignore[arg-type]
    lvls = mark_swept(htf_levels(df), df)
    mr = monday_range(df) or {}
    t = Table(title=f"{pair} liquidity levels")
    for col in ("label", "price", "ts", "swept"):
        t.add_column(col)
    for name, lvl in {**lvls, **mr}.items():
        t.add_row(name, f"{lvl.price:.5f}", lvl.ts.strftime("%Y-%m-%d %H:%M"),
                  "✔" if lvl.swept else "")
    print(t)


@app.command()
def sessions(
    pair: str = typer.Option("EURUSD"),
    tf: str = typer.Option("M15"),
    days: int = typer.Option(5),
    feed: str = typer.Option("duka"),
):
    """Show completed session H/L blocks (last N days)."""
    f = default_feed(feed)
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    df = f.history(pair, tf, start, end)  # type: ignore[arg-type]
    ranges = session_ranges(df)
    t = Table(title=f"{pair} sessions — {days}d")
    for col in ("date", "session", "high", "low", "range(pips)"):
        t.add_column(col)
    pip = 0.0001 if "JPY" not in pair else 0.01
    for r in ranges[-20:]:
        t.add_row(r.date.strftime("%Y-%m-%d"), r.session.value,
                  f"{r.high:.5f}", f"{r.low:.5f}",
                  f"{(r.high - r.low) / pip:.1f}")
    print(t)


@app.command()
def chart(
    pair: str = typer.Option("EURUSD"),
    tf: str = typer.Option("H1"),
    days: int = typer.Option(30),
    feed: str = typer.Option("duka"),
    swing_len: int = typer.Option(5),
    no_open: bool = typer.Option(False, help="Don't auto-open browser"),
):
    """Render an interactive HTML chart with swings, FVGs, OBs, levels, sessions."""
    from fsp.cli.chart import render
    out = render(pair, tf, days, feed, swing_len, open_browser=not no_open)
    print(f"[bold green]Chart written:[/] {out}")


@app.command()
def context(
    pair: str = typer.Option("EURUSD"),
    tf: str = typer.Option("H1"),
    days: int = typer.Option(30),
    feed: str = typer.Option("duka"),
):
    """Show current market cycle, H1 OF bias, and recent SMT divergences."""
    from fsp.context.cycle import classify_cycle
    from fsp.context.bias import compute_of_bias
    from fsp.context.smt import detect_smt_positive, detect_smt_negative

    f = default_feed(feed)
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    df = f.history(pair, tf, start, end)  # type: ignore[arg-type]
    daily = f.history(pair, "D", end - timedelta(days=30), end)  # type: ignore[arg-type]

    cyc = classify_cycle(df, daily)
    bias = compute_of_bias(df)

    print(f"\n[bold]Market Cycle ({pair} {tf})[/]")
    print(f"  Cycle:       [yellow]{cyc.cycle.value}[/]  (ATR ratio {cyc.atr_ratio:.2f})")
    print(f"  ATR fast/slow: {cyc.atr_fast:.5f} / {cyc.atr_slow:.5f}")
    print(f"  Today range / ADR(5): {cyc.today_range:.5f} / {cyc.adr5:.5f} → [bold]{cyc.adr_pct:.0f}%[/]")

    print(f"\n[bold]Order Flow Bias (H1 proxy)[/]")
    color = "green" if bias.bias.value == "BULL" else "red" if bias.bias.value == "BEAR" else "yellow"
    print(f"  Bias:       [{color}]{bias.bias.value}[/]")
    print(f"  Last event: {bias.last_event}")
    if bias.last_hh: print(f"  Last HH: {bias.last_hh:.5f}")
    if bias.last_hl: print(f"  Last HL: {bias.last_hl:.5f}")
    if bias.last_lh: print(f"  Last LH: {bias.last_lh:.5f}")
    if bias.last_ll: print(f"  Last LL: {bias.last_ll:.5f}")

    # SMT vs EUR/GBP (positive) and vs DXY (negative)
    print(f"\n[bold]SMT Divergences (last 10)[/]")
    try:
        other = "GBPUSD" if pair == "EURUSD" else "EURUSD"
        df_other = f.history(other, tf, start, end)  # type: ignore[arg-type]
        pos = detect_smt_positive(df, df_other, pair, other)
    except Exception as e:
        pos = []
        print(f"  [dim](positive-pair feed error: {e})[/]")

    try:
        # DXY isn't on Dukascopy → fall back to yfinance for this symbol only
        dxy_feed = default_feed("yf") if feed == "duka" else f
        df_dxy = dxy_feed.history("DXY", tf, start, end)  # type: ignore[arg-type]
        # Align timezones + intersect to overlap window
        neg = detect_smt_negative(df, df_dxy, pair, "DXY") if not df_dxy.empty else []
        if df_dxy.empty:
            print("  [dim](DXY feed returned no data — skipping negative SMT)[/]")
    except Exception as e:
        neg = []
        print(f"  [dim](DXY fetch failed: {type(e).__name__}: {e} — skipping negative SMT)[/]")

    combined = sorted(pos + neg, key=lambda e: e.ts)[-10:]
    if not combined:
        print("  [dim]No recent SMT events.[/]")
    for ev in combined:
        col = "green" if ev.kind == "bull" else "red"
        print(f"  [{col}]{ev.kind.upper()}[/] at {ev.at:>4} "
              f"{ev.ts:%Y-%m-%d %H:%M} · {ev.note}")


@app.command()
def grade(
    pair: str = typer.Option("EURUSD"),
    ltf: str = typer.Option("M15", help="Execution TF: M5 or M15"),
    feed: str = typer.Option("duka"),
    equity: float = typer.Option(10_000.0, help="Account equity for R sizing"),
    risk_pct: float = typer.Option(0.005, help="Risk per 1R as fraction of equity"),
):
    """Grade the CURRENT setup on `pair` using 4-Step Protocol checklist."""
    from fsp.grader.setup import grade_setup

    f = default_feed(feed)
    end = datetime.now(timezone.utc)
    ltf_df = f.history(pair, ltf, end - timedelta(days=5), end)     # type: ignore[arg-type]
    h1_df = f.history(pair, "H1", end - timedelta(days=30), end)    # type: ignore[arg-type]
    daily_df = f.history(pair, "D", end - timedelta(days=30), end)  # type: ignore[arg-type]

    other_pair = "GBPUSD" if pair == "EURUSD" else "EURUSD"
    try:
        other_df = f.history(other_pair, "H1", end - timedelta(days=30), end)  # type: ignore[arg-type]
    except Exception:
        other_df = None

    dxy_df = None
    try:
        dxy_feed = default_feed("yf") if feed == "duka" else f
        dxy_df = dxy_feed.history("DXY", "H1", end - timedelta(days=30), end)  # type: ignore[arg-type]
    except Exception:
        pass

    setup = grade_setup(pair, ltf_df, h1_df, daily_df,
                        other_df=other_df, other_pair=other_pair,
                        dxy_df=dxy_df,
                        account_equity=equity, base_risk_pct=risk_pct)

    # Render
    grade_color = {"A+": "green", "A": "green", "B": "yellow", "SKIP": "red"}[setup.grade.value]
    icon = {"A+": "🟢", "A": "🟢", "B": "🟡", "SKIP": "🔴"}[setup.grade.value]
    dir_str = setup.direction.upper() if setup.direction else "—"
    pip = 0.01 if "JPY" in pair else 0.0001

    print(f"\n{icon} [{grade_color}]SETUP — GRADE {setup.grade.value}[/]   {pair}  {ltf}  "
          f"[dim]{setup.context['now']}[/]")
    print(f"  Price:     {setup.context['price']:.5f}  "
          f"|  Session: {setup.context['session']}  |  Cycle: {setup.context['cycle']}  "
          f"|  ADR%: {setup.context['adr_pct']}  |  Bias: {setup.context['bias']} ({setup.context['bias_event']})")

    if setup.grade != Grade.SKIP and setup.entry is not None:
        dollar_risk = equity * risk_pct * setup.risk_r
        print(f"  Direction: [bold]{dir_str}[/]")
        print(f"  Key level: {setup.key_level_ref}")
        print(f"  Entry:     {setup.entry:.5f}")
        print(f"  SL:        {setup.sl:.5f}  ({setup.invalidation_pips:.1f} pips)")
        if setup.tp1 is not None:
            print(f"  TP1:       {setup.tp1:.5f}  ({setup.rr_tp1:.2f}R — {setup.context.get('opposing_target')})")
        if setup.tp2 is not None:
            print(f"  TP2:       {setup.tp2:.5f}  ({setup.rr_tp2:.2f}R)")
        print(f"  Risk:      [bold]{setup.risk_r:.1f}R[/]  = ${dollar_risk:.2f} (@ {risk_pct*100:.2f}% equity)")

    print(f"\n  Checklist {setup.passed()}/{setup.total()}")
    for c in setup.checklist:
        mark = "[green]✓[/]" if c.passed else "[red]✗[/]"
        print(f"    {mark} {c.name}" + (f"  [dim]({c.note})[/]" if c.note else ""))


@app.command("telegram-setup")
def telegram_setup():
    """Interactive wizard: save your Telegram bot token + chat ID to ~/.fsp/config.toml."""
    import asyncio
    from fsp.notify.config import load, save
    from fsp.notify.telegram import TelegramClient, get_updates_chat_id

    print("[bold]Telegram setup[/]")
    print("1. Open Telegram, search for [cyan]@BotFather[/], send [cyan]/newbot[/], follow prompts.")
    print("2. Copy the bot token it gives you (looks like 123456:ABC-DEF...).")
    print("3. Send any message to your new bot (search for it by the username you picked).\n")

    token = typer.prompt("Bot token").strip()
    print("\nFetching recent chats...")
    chats = asyncio.run(get_updates_chat_id(token))
    if not chats:
        print("[yellow]No chats found yet. Send any message to your bot in Telegram, then re-run this command.[/]")
        return

    print("\n[bold]Chats that have messaged the bot:[/]")
    for i, (cid, name) in enumerate(chats):
        print(f"  [{i}] {name}  (id={cid})")
    idx = int(typer.prompt("\nPick chat number") )
    chat_id = chats[idx][0]

    cfg = load()
    cfg["telegram"] = {"bot_token": token, "chat_id": chat_id}
    save(cfg)

    print(f"\n[green]Saved to ~/.fsp/config.toml[/]")
    tc = TelegramClient(token, chat_id)
    if tc.send_sync("✅ fsp connected. You'll receive setup alerts here."):
        print("[green]Test message delivered.[/]")
    else:
        print("[red]Test send failed — check token/chat id.[/]")


@app.command("live")
def live_cmd(
    pairs: str = typer.Option("EURUSD,GBPUSD,USDJPY,AUDUSD,USDCAD,EURJPY,GBPJPY", help="Comma-separated"),
    ltf: str = typer.Option("M15"),
    feed: str = typer.Option("duka", help="duka | yf | td (Twelve Data live)"),
    interval: int = typer.Option(300, help="Seconds between scans (default 5 min)"),
    min_grade: str = typer.Option("B", help="A+ | A | B"),
    equity: float = typer.Option(10_000.0),
    risk_pct: float = typer.Option(0.005),
    dry: bool = typer.Option(False, help="Don't send Telegram, just print"),
    llm: bool = typer.Option(False, help="Enable LLM signal validation via Bedrock"),
):
    """Run the always-on signal monitor. Sends Telegram alerts on qualifying setups."""
    import asyncio
    from fsp.notify.live import live_loop
    grade_map = {"A+": Grade.A_PLUS, "A": Grade.A, "B": Grade.B}
    mg = grade_map[min_grade]
    pair_list = [p.strip().upper() for p in pairs.split(",") if p.strip()]
    try:
        asyncio.run(live_loop(pair_list, ltf, feed, interval, mg, equity, risk_pct, dry, use_llm=llm))
    except KeyboardInterrupt:
        print("\n[yellow]stopped[/]")


@app.command()
def journal(limit: int = typer.Option(20)):
    """Show the last N logged signals."""
    import sqlite3, json
    from fsp.journal.db import DB_PATH
    if not DB_PATH.exists():
        print("[dim]No journal yet.[/]"); return
    c = sqlite3.connect(DB_PATH)
    rows = c.execute("SELECT ts, pair, grade, direction, entry, sl, tp1, risk_r, sent_to_telegram "
                     "FROM signals ORDER BY ts DESC LIMIT ?", (limit,)).fetchall()
    t = Table(title=f"Last {len(rows)} signals")
    for col in ("ts", "pair", "grade", "dir", "entry", "sl", "tp1", "R", "sent"):
        t.add_column(col)
    for r in rows:
        t.add_row(r[0][:19], r[1], r[2], r[3] or "-",
                  f"{r[4]:.5f}" if r[4] else "-",
                  f"{r[5]:.5f}" if r[5] else "-",
                  f"{r[6]:.5f}" if r[6] else "-",
                  f"{r[7]:.1f}" if r[7] else "-",
                  "✔" if r[8] else "")
    print(t)


@app.command()
def backtest(
    pair: str = typer.Option("EURUSD"),
    start: str = typer.Option(..., help="YYYY-MM-DD"),
    end: str = typer.Option(..., help="YYYY-MM-DD"),
    ltf: str = typer.Option("M15"),
    min_grade: str = typer.Option("A", help="A+ | A | B"),
    feed: str = typer.Option("duka"),
    stride: int = typer.Option(1, help="Check every Nth LTF bar (speed vs accuracy)"),
    include_dxy: bool = typer.Option(False),
    no_open: bool = typer.Option(False),
):
    """Backtest the grader on historical data, produce HTML report."""
    from fsp.backtest.engine import run_backtest
    from fsp.backtest.report import render

    grade_map = {"A+": Grade.A_PLUS, "A": Grade.A, "B": Grade.B}
    s = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
    e = datetime.fromisoformat(end).replace(tzinfo=timezone.utc)

    print(f"[cyan]Backtesting {pair} {ltf} {start}→{end} min_grade={min_grade} stride={stride}[/]")
    res = run_backtest(pair, s, e, ltf=ltf, feed_kind=feed,
                       min_grade=grade_map[min_grade], stride=stride,
                       include_dxy=include_dxy)
    st = res.stats()
    print(f"\n[bold]Done:[/] {st.get('total', 0)} trades")
    if st.get("total", 0) > 0:
        print(f"  Win rate:     {st['win_rate']*100:.1f}% ({st['wins']}/{st['total']})")
        print(f"  Expectancy:   {st['expectancy']:+.3f}R / trade")
        print(f"  Profit factor: {st['profit_factor']:.2f}")
        print(f"  Total R:      {st['total_r']:+.1f}")
        print(f"  Max drawdown: {st['max_dd']:.1f}R")
        print(f"  By grade:     {st['by_grade']}")

    label = f"{start}_{end}_{min_grade.replace('+','p')}_s{stride}"
    out = render(res, pair, label)
    print(f"\n[green]Report:[/] {out}")
    if not no_open:
        import webbrowser
        webbrowser.open(f"file://{out}")


@app.command()
def diagnose(
    pair: str = typer.Option("EURUSD"),
    start: str = typer.Option(..., help="YYYY-MM-DD"),
    end: str = typer.Option(..., help="YYYY-MM-DD"),
    ltf: str = typer.Option("M15"),
    min_grade: str = typer.Option("A"),
    stride: int = typer.Option(4),
):
    """Run backtest + correlate each checklist item with win/loss."""
    from datetime import datetime, timezone
    from fsp.backtest.diagnose import rebuild_diagnostics, checklist_correlation, slice_stats

    s = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
    e = datetime.fromisoformat(end).replace(tzinfo=timezone.utc)
    grade_map = {"A+": Grade.A_PLUS, "A": Grade.A, "B": Grade.B}

    print(f"[cyan]Diagnosing {pair} {start}→{end} min_grade={min_grade} stride={stride}[/]")
    res, diags = rebuild_diagnostics(pair, s, e, ltf=ltf, min_grade=grade_map[min_grade], stride=stride)

    st = res.stats()
    n_closed = st.get("total", 0)
    print(f"\n[bold]Population:[/] {len(res.trades)} closed trades ({len(diags)} with checklist)")
    if n_closed == 0:
        print("[red]No trades — nothing to diagnose.[/]"); return

    print(f"  Win rate: {st['win_rate']*100:.1f}%  Exp: {st['expectancy']:+.3f}R  PF: {st['profit_factor']:.2f}")
    print(f"  Avg win/loss: {st['avg_win']:+.2f} / {st['avg_loss']:+.2f}  (ratio {st['avg_win_over_loss']:.2f})")
    print(f"  Outcomes: {st['by_outcome']}")

    # ---- Checklist item correlation ----
    corr = checklist_correlation(diags)
    t = Table(title="Checklist item predictive lift (pass_wr - fail_wr)")
    for c in ("item", "pass_n", "pass_wr", "fail_n", "fail_wr", "lift_pp"):
        t.add_column(c)
    for row in corr:
        color = "green" if row["lift_pp"] > 10 else "yellow" if row["lift_pp"] > 0 else "red"
        t.add_row(row["item"][:38], str(row["pass_n"]),
                  f"{row['pass_wr']:.0f}%", str(row["fail_n"]), f"{row['fail_wr']:.0f}%",
                  f"[{color}]{row['lift_pp']:+.0f}pp[/]")
    print(t)

    # ---- By session / direction / day-of-week / cycle ----
    for label, keyfn in [
        ("session", lambda d: d.session),
        ("direction", lambda d: d.direction),
        ("day-of-week", lambda d: d.dow),
        ("cycle", lambda d: d.cycle),
        ("bias_event", lambda d: d.bias_event),
    ]:
        rows = slice_stats(diags, keyfn, min_n=3)
        if not rows:
            continue
        tt = Table(title=f"Performance by {label}")
        for col in ("key", "n", "wr", "exp"):
            tt.add_column(col)
        for r in rows:
            tt.add_row(r["key"], str(r["n"]), f"{r['wr']:.0f}%", f"{r['exp']:+.2f}R")
        print(tt)

    # ---- ADR% bucketing ----
    adr_buckets = {"<50%": [], "50-100%": [], "100-150%": [], ">150%": []}
    for d in diags:
        if d.adr_pct < 50: adr_buckets["<50%"].append(d)
        elif d.adr_pct < 100: adr_buckets["50-100%"].append(d)
        elif d.adr_pct < 150: adr_buckets["100-150%"].append(d)
        else: adr_buckets[">150%"].append(d)
    tt = Table(title="Performance by ADR% at entry")
    for col in ("bucket", "n", "wr", "exp"):
        tt.add_column(col)
    for k, group in adr_buckets.items():
        if not group: continue
        wins = sum(1 for g in group if g.won)
        r = sum(g.weighted_r for g in group)
        tt.add_row(k, str(len(group)), f"{wins/len(group)*100:.0f}%", f"{r/len(group):+.2f}R")
    print(tt)

    # ---- Worst 10 trades detail ----
    closed = [t for t in res.trades if t.outcome not in ("open", "pending")]
    worst = sorted(closed, key=lambda t: t.weighted_r)[:10]
    tt = Table(title="Worst 10 trades")
    for col in ("open_ts", "dir", "grade", "session", "outcome", "R", "bars"):
        tt.add_column(col)
    for t in worst:
        tt.add_row(str(t.open_ts)[:16], t.direction, t.grade, t.session,
                   t.outcome, f"{t.weighted_r:+.2f}", str(t.bars_held))
    print(tt)


@app.command("signals")
def signals_cmd(
    pair: str = typer.Option("EURUSD,GBPUSD,USDJPY,AUDUSD,USDCAD,EURJPY,GBPJPY", help="Comma-separated"),
    feed: str = typer.Option("yf", help="duka | yf  (yf is faster for recent data)"),
    dry: bool = typer.Option(True, help="Print only — don't send Telegram"),
):
    """Run the intraday signal scanner once and show results."""
    import asyncio
    from fsp.signals.scanner import scan_pair_live

    pair_list = [p.strip().upper() for p in pair.split(",") if p.strip()]

    async def _run():
        for p in pair_list:
            print(f"\n[cyan]Scanning {p}...[/]")
            sigs = await scan_pair_live(p, feed)
            if not sigs:
                print(f"  [dim]No signals right now.[/]")
            for sig in sigs:
                icon = "🔵" if sig.direction == "long" else "🟠"
                print(f"  {icon} [{sig.strategy}] {sig.direction.upper()} "
                      f"entry={sig.entry:.5f} sl={sig.sl:.5f} "
                      f"tp1={sig.tp1:.5f} ({sig.rr_tp1:.1f}R)  {sig.note}")
            if not dry and sigs:
                from fsp.notify.config import load as load_cfg
                from fsp.notify.telegram import TelegramClient, format_signal
                from fsp.journal.db import log_intraday_signal
                cfg = load_cfg()
                tg_cfg = cfg.get("telegram", {})
                if tg_cfg.get("bot_token") and tg_cfg.get("chat_id"):
                    tg = TelegramClient(tg_cfg["bot_token"], tg_cfg["chat_id"])
                    for sig in sigs:
                        ok = await tg.send(format_signal(sig))
                        log_intraday_signal(sig, sig.dedup_key, sent=ok)
                        print(f"  [green]Sent {'✓' if ok else '✗'}[/]")

    asyncio.run(_run())


@app.command("intraday-journal")
def intraday_journal_cmd(limit: int = typer.Option(30)):
    """Show the last N intraday signals (ECM + ARB) from the journal."""
    import sqlite3
    from fsp.journal.db import DB_PATH
    if not DB_PATH.exists():
        print("[dim]No journal yet.[/]")
        return
    c = sqlite3.connect(DB_PATH)
    try:
        rows = c.execute(
            "SELECT ts, pair, strategy, direction, entry, sl, tp1, rr_tp1, "
            "inv_pips, risk_r, note, sent_to_telegram, outcome "
            "FROM intraday_signals ORDER BY ts DESC LIMIT ?", (limit,)
        ).fetchall()
    except sqlite3.OperationalError:
        print("[dim]No intraday_signals table yet — run `fsp live` first.[/]")
        return
    if not rows:
        print("[dim]No intraday signals logged yet.[/]")
        return
    t = Table(title=f"Last {len(rows)} intraday signals")
    for col in ("ts", "pair", "strat", "dir", "entry", "sl", "tp1", "RR", "pips", "note", "sent", "outcome"):
        t.add_column(col)
    for r in rows:
        t.add_row(
            r[0][:16], r[1], r[2], r[3] or "-",
            f"{r[4]:.5f}" if r[4] else "-",
            f"{r[5]:.5f}" if r[5] else "-",
            f"{r[6]:.5f}" if r[6] else "-",
            f"{r[7]:.1f}" if r[7] else "-",
            f"{r[8]:.1f}" if r[8] else "-",
            (r[10] or "")[:40],
            "✔" if r[11] else "",
            r[12] or "",
        )
    print(t)


@app.command("intraday-backtest")
def intraday_backtest_cmd(
    pair: str = typer.Option("EURUSD", help="EURUSD | GBPUSD"),
    start: str = typer.Option(..., help="YYYY-MM-DD"),
    end: str = typer.Option(..., help="YYYY-MM-DD"),
    strategies: str = typer.Option("ECM,ARB", help="Comma-separated: ECM,ARB"),
    feed: str = typer.Option("duka", help="duka | yf"),
    stride: int = typer.Option(1, help="Check every Nth bar (1=full, 3=3× faster)"),
    no_open: bool = typer.Option(False),
    verbose: bool = typer.Option(False),
):
    """Backtest intraday strategies (EMA Cross Momentum + Asian Range Breakout)."""
    from fsp.backtest.intraday_engine import run_intraday_backtest
    from fsp.backtest.intraday_report import render_intraday

    strats = [s.strip().upper() for s in strategies.split(",") if s.strip()]
    s_dt = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
    e_dt = datetime.fromisoformat(end).replace(tzinfo=timezone.utc)

    print(f"[cyan]Intraday backtest[/] · {pair} {start}→{end} strategies={strats} stride={stride}")
    results = run_intraday_backtest(
        pair, s_dt, e_dt,
        strategies=strats,
        feed_kind=feed,
        stride_m15=stride,
        stride_m5=stride,
        verbose=verbose,
    )

    label = f"{start}_{end}_s{stride}"
    for strat, res in results.items():
        st = res.stats()
        n = st.get("total", 0)
        print(f"\n[bold]{strat}[/] — {n} trades")
        if n > 0:
            print(f"  Win rate:    {st['win_rate']*100:.1f}%  ({st['wins']}/{n})")
            print(f"  Expectancy:  {st['expectancy']:+.3f}R / trade")
            print(f"  PF:          {st['profit_factor']:.2f}")
            print(f"  Total R:     {st['total_r']:+.1f}R")
            print(f"  Max DD:      {st['max_dd']:.1f}R")
            print(f"  By session:  {st.get('by_session', {})}")
            print(f"  By DOW:      {st.get('by_dow', {})}")
            print(f"  Outcomes:    {st.get('by_outcome', {})}")
        else:
            print("  [dim](no trades fired)[/]")

    out = render_intraday(results, pair, label)
    print(f"\n[green]Report:[/] {out}")
    if not no_open:
        import webbrowser
        webbrowser.open(f"file://{out}")


@app.command("resolve-outcomes")
def resolve_outcomes_cmd(
    verbose: bool = typer.Option(True, help="Print each resolved signal"),
):
    """Walk unresolved journal signals against M15 price data and mark win/loss/timeout.

    Run this after each trading session (or daily) to keep the journal current.
    Signals fired within the last ~2 hours may still be 'open' — re-run later.
    """
    from fsp.journal.resolver import resolve_all
    print("[cyan]Resolving outcomes...[/]")
    result = resolve_all(verbose=verbose)
    print(
        f"\n[green]Done.[/]  resolved={result['resolved']}  "
        f"skipped={result['skipped']} (too recent)  errors={result['errors']}"
    )


@app.command("review")
def review_cmd(
    strategy: str = typer.Option("TREND_RSI", help="Strategy to review"),
):
    """Show the learning report: WR/Expectancy by RSI depth, session, DOW, pair.

    Run `fsp resolve-outcomes` first to make sure all outcomes are filled in.
    """
    from fsp.journal.review import build_report
    print(build_report(strategy))
