"""`fsp live` loop — grades 4SP setups AND scans intraday strategies every interval."""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone

from rich import print

from fsp.data.feed import default_feed
from fsp.data.types import Grade
from fsp.grader.setup import grade_setup, SetupCandidate
from fsp.journal.db import last_signal_dedup_key, log_signal, log_intraday_signal
from fsp.notify.config import load as load_cfg
from fsp.notify.telegram import TelegramClient, format_setup, format_signal
from fsp.signals.scanner import scan_pair_live

log = logging.getLogger("fsp.live")


# ── 4SP helpers ──────────────────────────────────────────────────────────────

def _dedup_key(s: SetupCandidate) -> str:
    if s.grade == Grade.SKIP or s.entry is None:
        return f"{s.pair}|SKIP"
    return f"{s.pair}|{s.grade.value}|{s.direction}|{s.entry:.5f}|{s.sl:.5f}"


async def _grade_once(pair: str, ltf: str, feed_kind: str, equity: float,
                      risk_pct: float) -> SetupCandidate:
    f = default_feed(feed_kind)
    end = datetime.now(timezone.utc)
    ltf_df = f.history(pair, ltf, end - timedelta(days=5), end)
    h1_df = f.history(pair, "H1", end - timedelta(days=30), end)
    daily_df = f.history(pair, "D", end - timedelta(days=30), end)
    other = "GBPUSD" if pair == "EURUSD" else "EURUSD"
    other_df = None
    try:
        other_df = f.history(other, "H1", end - timedelta(days=30), end)
    except Exception:
        pass
    dxy_df = None
    try:
        dxy_df = default_feed("yf").history("DXY", "H1", end - timedelta(days=30), end)
    except Exception:
        pass
    return grade_setup(pair, ltf_df, h1_df, daily_df,
                       other_df=other_df, other_pair=other,
                       dxy_df=dxy_df, account_equity=equity, base_risk_pct=risk_pct)


# ── Main loop ─────────────────────────────────────────────────────────────────

async def live_loop(pairs: list[str], ltf: str, feed_kind: str,
                    interval_sec: int, min_grade: Grade,
                    equity: float, risk_pct: float, dry: bool):
    cfg = load_cfg()
    tg_cfg = cfg.get("telegram", {})
    tg = None
    if not dry:
        if not tg_cfg.get("bot_token") or not tg_cfg.get("chat_id"):
            print("[red]No telegram config — run `fsp telegram-setup` or use --dry.[/]")
            return
        tg = TelegramClient(tg_cfg["bot_token"], tg_cfg["chat_id"])

    grade_rank = {Grade.SKIP: 0, Grade.B: 1, Grade.A: 2, Grade.A_PLUS: 3}
    min_rank = grade_rank[min_grade]

    print(f"[cyan]fsp live[/] · pairs={pairs} ltf={ltf} interval={interval_sec}s "
          f"min_grade={min_grade.value} dry={dry}")
    print("[dim]Running: 4-Step Protocol grader + EMA Cross Momentum + Asian Range Breakout[/]")

    while True:
        t0 = datetime.now(timezone.utc)

        for pair in pairs:
            # ── 4SP grader ──────────────────────────────────────
            try:
                s = await _grade_once(pair, ltf, feed_kind, equity, risk_pct)
                key = _dedup_key(s)
                last = last_signal_dedup_key(pair, minutes=60)
                repeat = key == last
                rank = grade_rank[s.grade]
                should_send = rank >= min_rank and not repeat and not dry

                line = (f"{t0:%H:%M:%S} [bold]{pair}[/] [4SP] {s.grade.value}"
                        f" {s.direction or '-'}"
                        f" {s.passed()}/{s.total()}"
                        f" {'(dup)' if repeat else ''}")
                print(line)

                if should_send and tg:
                    ok = await tg.send(format_setup(s))
                    log_signal(s, key, sent=ok)
                else:
                    log_signal(s, key, sent=False)

            except Exception as e:
                log.exception("4SP grade failed %s", pair)
                print(f"[red]{pair} [4SP]: {type(e).__name__}: {e}[/]")

            # ── Intraday signal scanner ──────────────────────────
            try:
                signals = await scan_pair_live(pair, feed_kind)
                for sig in signals:
                    dk = sig.dedup_key
                    last_sig = last_signal_dedup_key(pair, minutes=120,
                                                     strategy=sig.strategy)
                    # Compare pair|strategy|direction only — ignore entry price
                    # (entry changes every bar but it is still the same setup)
                    dk_core   = "|".join(dk.split("|")[:3])
                    last_core = "|".join(last_sig.split("|")[:3]) if last_sig else ""
                    repeat_sig = dk_core == last_core

                    print(f"{t0:%H:%M:%S} [bold]{pair}[/] [{sig.strategy}] "
                          f"{sig.direction.upper()} entry={sig.entry:.5f} "
                          f"rr={sig.rr_tp1:.1f} "
                          f"{'(dup)' if repeat_sig else ''}")

                    if not repeat_sig and not dry and tg:
                        ok = await tg.send(format_signal(sig))
                        log_intraday_signal(sig, dk, sent=ok)
                    else:
                        log_intraday_signal(sig, dk, sent=False)

            except Exception as e:
                log.exception("Intraday scan failed %s", pair)
                print(f"[red]{pair} [signals]: {type(e).__name__}: {e}[/]")

        elapsed = (datetime.now(timezone.utc) - t0).total_seconds()
        sleep_for = max(5, interval_sec - elapsed)
        await asyncio.sleep(sleep_for)



