"""`fsp live` loop — grades 4SP setups AND scans intraday strategies every interval."""
from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timedelta, timezone

from rich import print

from fsp.data.feed import default_feed
from fsp.data.types import Grade
from fsp.grader.setup import grade_setup, SetupCandidate
from fsp.journal.db import (
    last_signal_dedup_key, log_signal, log_intraday_signal, update_features,
)
from fsp.notify.config import load as load_cfg, parse_chat_ids
from fsp.notify.chat import ChatHandler
from fsp.notify.daily_report import REPORT_HOUR_UTC, send_daily_report, should_send_report
from fsp.notify.telegram import TelegramClient, escape_md, format_setup, format_signal
from fsp.signals.scanner import scan_pair_live, scan_batch_live

log = logging.getLogger("fsp.live")


# ── Watchdog (kills frozen scan loops) ──────────────────────────────────────
WATCHDOG_WARN_MIN = float(os.environ.get("FSP_WATCHDOG_WARN_MIN", "10"))
WATCHDOG_KILL_MIN = float(os.environ.get("FSP_WATCHDOG_KILL_MIN", "20"))

# How often to resolve journalled signal outcomes (seconds). Hourly by default.
RESOLVE_INTERVAL_SEC = float(os.environ.get("FSP_RESOLVE_INTERVAL_SEC", "3600"))


async def _watchdog_loop(cycle_ref: dict, tg: TelegramClient | None) -> None:
    """Monitor cycle heartbeat. Alert at WARN, force-restart at KILL.

    The scanner has historically frozen silently after 12-24h of operation
    (yfinance/network hangs Fargate cant detect). Watchdog ensures any stall
    longer than WATCHDOG_KILL_MIN forces ECS to restart the task.
    """
    warned = False
    while True:
        await asyncio.sleep(60)  # check every minute
        last = cycle_ref.get("last_cycle_at")
        if last is None:
            continue  # first cycle hasnt completed yet
        elapsed = (datetime.now(timezone.utc) - last).total_seconds()

        if elapsed > WATCHDOG_KILL_MIN * 60:
            msg = (f"🚨 *Watchdog*: scan loop frozen for {elapsed/60:.0f} min "
                   f"(>{WATCHDOG_KILL_MIN}). Forcing restart.")
            log.error(msg)
            if tg:
                try:
                    await asyncio.wait_for(tg.send(msg), timeout=10)
                except Exception:
                    pass
            os._exit(1)  # Fargate will restart the task

        elif elapsed > WATCHDOG_WARN_MIN * 60 and not warned:
            warned = True
            msg = (f"⚠️ *Watchdog*: scan loop stalled for {elapsed/60:.0f} min "
                   f"(threshold {WATCHDOG_WARN_MIN}). Will force-restart at "
                   f"{WATCHDOG_KILL_MIN} min.")
            log.warning(msg)
            if tg:
                try:
                    await asyncio.wait_for(tg.send(msg), timeout=10)
                except Exception:
                    pass

        elif elapsed < WATCHDOG_WARN_MIN * 60 and warned:
            warned = False  # cycles resumed — reset for next stall


async def _run_resolver_bg() -> None:
    """Resolve journalled signal outcomes off the critical path.

    Runs ``resolve_all`` in a worker thread so a slow/hanging yfinance fetch
    can't stall the scan loop (and trip the watchdog). Fire-and-forget;
    logs a one-line summary.
    """
    try:
        from fsp.journal.resolver import resolve_all
        counts = await asyncio.to_thread(resolve_all, False)
        log.info("resolver: %s", counts)
        print(f"[cyan]🧮 Resolver: {counts['resolved']} resolved, "
              f"{counts['skipped']} skipped, {counts['errors']} errors[/]")
    except Exception as e:
        log.exception("Background resolver failed")
        print(f"[red]Resolver error: {type(e).__name__}: {e}[/]")


# ── ML feature extractor (lazy-loaded) ────────────────────────────────────────
_feature_extractor = None


def _get_feature_extractor():
    global _feature_extractor
    if _feature_extractor is None:
        try:
            from fsp.ml.features import FeatureExtractor
            _feature_extractor = FeatureExtractor()
            log.info("ML feature extractor initialised")
        except Exception as e:
            log.warning("Feature extractor unavailable: %s", e)
            _feature_extractor = False
    return _feature_extractor if _feature_extractor else None


async def _extract_and_persist_features(
    sig, signal_id: int, llm_result, feed_kind: str,
) -> None:
    """Best-effort feature extraction. Failures never break the signal flow."""
    try:
        fe = _get_feature_extractor()
        if fe is None:
            return
        f = default_feed(feed_kind)
        end = datetime.now(timezone.utc)
        m15 = f.history(sig.pair, "M15", end - timedelta(days=5), end)
        h1 = f.history(sig.pair, "H1", end - timedelta(days=30), end)
        # Build H4 from M15 (consistent with scanner)
        h4 = None
        try:
            if len(m15) >= 55:
                h4 = m15.resample("4h").agg({
                    "open": "first", "high": "max", "low": "min",
                    "close": "last", "volume": "sum",
                }).dropna()
        except Exception:
            pass
        ts = datetime.fromisoformat(sig.ts) if isinstance(sig.ts, str) else sig.ts
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        fs = fe.extract(
            pair=sig.pair, strategy=sig.strategy, direction=sig.direction,
            ts=ts, m15=m15, h1=h1, h4=h4, llm_result=llm_result,
        )
        update_features(signal_id, fs.to_dict())
        log.info("Features extracted for signal %d", signal_id)
    except Exception as e:
        log.warning("Feature extraction failed for signal %s: %s",
                    getattr(sig, "dedup_key", "?"), e)


# ── LLM Validator (lazy-loaded) ──────────────────────────────────────────────
_validator = None


def _get_validator():
    """Lazy-init the Bedrock signal validator."""
    global _validator
    if _validator is None:
        try:
            from fsp.llm.validator import SignalValidator
            _validator = SignalValidator()
            log.info("LLM signal validator initialised (Bedrock)")
        except Exception as e:
            log.warning("LLM validator unavailable: %s", e)
            _validator = False  # sentinel: tried and failed
    return _validator if _validator else None


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
                    equity: float, risk_pct: float, dry: bool,
                    use_llm: bool = False):
    cfg = load_cfg()
    tg_cfg = cfg.get("telegram", {})
    tg = None
    if not dry:
        if not tg_cfg.get("bot_token") or not tg_cfg.get("chat_id"):
            print("[red]No telegram config — run `fsp telegram-setup` or use --dry.[/]")
            return
        # chat_id may be comma-separated: "primary,extra1,extra2"
        # Primary = your DM (gets commands). Extras = fan-out only (groups, etc).
        primary, extras = parse_chat_ids(tg_cfg["chat_id"])
        tg = TelegramClient(tg_cfg["bot_token"], primary, extras)
        if extras:
            print(f"[cyan]📡 Telegram fan-out: primary={primary}, extras={extras}[/]")

    grade_rank = {Grade.SKIP: 0, Grade.B: 1, Grade.A: 2, Grade.A_PLUS: 3}
    min_rank = grade_rank[min_grade]

    # Skip 4SP grader if using rate-limited feed (saves ~18 API calls/cycle)
    skip_4sp = feed_kind == "td"

    print(f"[cyan]fsp live[/] · pairs={pairs} ltf={ltf} interval={interval_sec}s "
          f"min_grade={min_grade.value} dry={dry} llm={use_llm}")
    if skip_4sp:
        print("[dim]Running: intraday scanner only (4SP skipped - rate-limited feed)[/]")
    else:
        print("[dim]Running: 4-Step Protocol grader + intraday scanner[/]")

    cycle = 0
    last_report_date: str | None = None
    last_resolve_at: datetime | None = None
    resolve_task: asyncio.Task | None = None

    # ── Spawn chat poller as background task ─────────────────────────────────
    cycle_ref: dict = {"cycle": 0, "interval_sec": interval_sec, "last_cycle_at": None}
    chat_task = None
    if tg:
        chat_handler = ChatHandler(tg, cycle_ref=cycle_ref)
        chat_task = asyncio.create_task(chat_handler.poll_loop())
        print("[cyan]💬 Chat poller started — send /help in Telegram to interact[/]")

    # ── Watchdog: detects frozen loops + forces restart on stall ────────────
    watchdog_task = asyncio.create_task(_watchdog_loop(cycle_ref, tg))
    print(f"[cyan]🐕 Watchdog active — warn at {WATCHDOG_WARN_MIN}min, "
          f"kill at {WATCHDOG_KILL_MIN}min[/]")

    while True:
        cycle += 1
        cycle_ref["cycle"] = cycle
        t0 = datetime.now(timezone.utc)
        total_signals = 0
        total_sent = 0

        # ── Daily report (once per UTC day, at REPORT_HOUR_UTC) ────────────────
        if tg and should_send_report(t0, last_report_date):
            try:
                ok = await send_daily_report(
                    tg,
                    hours=24,
                    pair=pairs[0] if pairs else "USDCAD",
                    cycle_count=cycle,
                    interval_sec=interval_sec,
                )
                if ok:
                    last_report_date = t0.strftime("%Y-%m-%d")
                    print(f"[green]📊 Daily report sent at {t0:%H:%M} UTC[/]")
                else:
                    print(f"[yellow]⚠ Daily report send failed (will retry next cycle)[/]")
            except Exception as e:
                log.exception("Daily report failed")
                print(f"[red]Daily report error: {type(e).__name__}: {e}[/]")

        # ── Outcome resolver (hourly, off the critical path) ───────────
        # Stamps real forward outcomes on journalled signals (incl ICT_SHADOW).
        # Runs in a worker thread so a slow yfinance fetch can't stall the
        # scan loop or trip the watchdog.
        if last_resolve_at is None or \
                (t0 - last_resolve_at).total_seconds() >= RESOLVE_INTERVAL_SEC:
            last_resolve_at = t0
            resolve_task = asyncio.create_task(_run_resolver_bg())

        # ── Batch fetch all pairs (4 API calls instead of 24) ────
        # Wrap in cycle-level timeout — even with per-call yf timeouts, cumulative
        # slowness across 28 sequential calls can exceed the cycle interval.
        # Cap the entire batch at 2x the interval; abort if exceeded.
        batch_result = None
        try:
            batch_timeout = max(120.0, interval_sec * 2)
            batch_result = await asyncio.wait_for(
                scan_batch_live(pairs, feed_kind), timeout=batch_timeout,
            )
        except asyncio.TimeoutError:
            log.error("Batch scan exceeded %.0fs — abandoning cycle", batch_timeout)
            print(f"[red]⚠ Batch scan timed out after {batch_timeout:.0f}s — skipping cycle[/]")
            cycle_ref["last_cycle_at"] = datetime.now(timezone.utc)  # heartbeat so watchdog doesnt fire
            await asyncio.sleep(30)
            continue
        except Exception as e:
            err_msg = str(e)
            # Per-minute rate limit: short pause, retry next cycle
            if "current minute" in err_msg:
                print(f"[yellow]⚠ Per-minute rate limit hit — pausing 60s[/]")
                await asyncio.sleep(60)
                continue
            # Daily quota exhausted: long pause until midnight UTC
            if "run out of API credits" in err_msg:
                from datetime import datetime as dt
                now = dt.now(timezone.utc)
                tomorrow = (now + timedelta(days=1)).replace(
                    hour=0, minute=5, second=0, microsecond=0)
                pause_secs = (tomorrow - now).total_seconds()
                print(f"[yellow]⚠ Daily API credits exhausted — pausing until {tomorrow:%H:%M} UTC "
                      f"({pause_secs/3600:.1f}h)[/]")
                await asyncio.sleep(pause_secs)
                continue
            log.exception("Batch scan failed")
            print(f"[red]Batch scan error: {type(e).__name__}: {e}[/]")

        for pair in pairs:
            # ── 4SP grader (skipped for rate-limited feeds) ─────
            if not skip_4sp:
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

            # ── Intraday signal scanner (uses batch data) ────────
            if batch_result is None:
                continue
            try:
                signals = batch_result.get(pair, [])
                if not signals:
                    print(f"{t0:%H:%M:%S} [dim]{pair}[/] — no signals")
                total_signals += len(signals)
                for sig in signals:
                    dk = sig.dedup_key
                    last_sig = last_signal_dedup_key(pair, minutes=120,
                                                     strategy=sig.strategy)
                    # Compare pair|strategy|direction only — ignore entry price
                    # (entry changes every bar but it is still the same setup)
                    dk_core   = "|".join(dk.split("|")[:3])
                    last_core = "|".join(last_sig.split("|")[:3]) if last_sig else ""
                    repeat_sig = dk_core == last_core

                    # ICT_SHADOW: journal-only, never alert/trade
                    if sig.strategy == "ICT_SHADOW":
                        if not repeat_sig:
                            log_intraday_signal(sig, dk, sent=False)
                            print(f"{t0:%H:%M:%S} [magenta]{pair}[/] [ICT_SHADOW] "
                                  f"{sig.direction.upper()} entry={sig.entry:.5f} "
                                  f"rr={sig.rr_tp1:.1f} — {sig.note} (logged)")
                        continue

                    print(f"{t0:%H:%M:%S} [bold]{pair}[/] [{sig.strategy}] "
                          f"{sig.direction.upper()} entry={sig.entry:.5f} "
                          f"rr={sig.rr_tp1:.1f} "
                          f"{'(dup)' if repeat_sig else ''}")

                    if not repeat_sig and not dry and tg:
                        # LLM analyst review
                        llm_result = None
                        if use_llm:
                            v = _get_validator()
                            if v:
                                try:
                                    from fsp.llm.context import build_context
                                    ctx = build_context(pair, sig.strategy)
                                    llm_result = v.validate(
                                        pair=pair,
                                        direction=sig.direction,
                                        strategy=sig.strategy,
                                        entry=sig.entry,
                                        sl=sig.sl,
                                        tp=sig.tp1,
                                        context=ctx,
                                    )
                                    print(f"  [dim]LLM ({llm_result.model_used.split('.')[-1]}): "
                                          f"{llm_result.decision} ({llm_result.confidence:.0%}) — "
                                          f"{llm_result.reason}[/]")
                                except Exception as e:
                                    log.warning("LLM analyst error: %s", e)

                        decision = llm_result.decision if llm_result else "TAKE"

                        if decision == "SKIP":
                            print(f"  [yellow]SKIPPED by LLM: {llm_result.reason}[/]")
                            sig_id = log_intraday_signal(sig, dk, sent=False)
                        else:
                            msg = format_signal(sig)
                            # Analyst reasoning (escape Markdown special chars in LLM text)
                            if llm_result and llm_result.analysis:
                                msg += f"\n\n🧠 *Analyst:* {escape_md(llm_result.analysis)}"
                            # Suggested level modifications
                            if llm_result and llm_result.suggested_sl:
                                msg += f"\n📍 Suggested SL: `{llm_result.suggested_sl:.5f}`"
                            if llm_result and llm_result.suggested_tp:
                                msg += f"\n🎯 Suggested TP: `{llm_result.suggested_tp:.5f}`"
                            if decision == "REDUCE":
                                msg += "\n\n⚠️ *REDUCE to half position*"
                            ok = await tg.send(msg)
                            sig_id = log_intraday_signal(sig, dk, sent=ok)

                        # Extract + persist features for ML training (best-effort)
                        if sig_id:
                            asyncio.create_task(_extract_and_persist_features(
                                sig, sig_id, llm_result, feed_kind,
                            ))
                    else:
                        log_intraday_signal(sig, dk, sent=False)

            except Exception as e:
                log.exception("Intraday scan failed %s", pair)
                print(f"[red]{pair} [signals]: {type(e).__name__}: {e}[/]")

        elapsed = (datetime.now(timezone.utc) - t0).total_seconds()
        sleep_for = max(5, interval_sec - elapsed)
        # Record heartbeat for watchdog — must be set on EVERY cycle.
        cycle_ref["last_cycle_at"] = datetime.now(timezone.utc)
        print(f"[cyan]── cycle {cycle} done[/] · {len(pairs)} pairs · "
              f"{total_signals} signals · scan={elapsed:.0f}s · "
              f"sleep={sleep_for:.0f}s")
        await asyncio.sleep(sleep_for)



