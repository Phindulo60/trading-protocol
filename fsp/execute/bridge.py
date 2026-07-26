"""FSP -> mt4-executor execution bridge.

At the point the live loop decides to *alert* a signal (passed dedup, grade and
the LLM TAKE/REDUCE gate), ``maybe_execute`` can also enqueue a live order by
inserting one row into the Supabase ``commands`` table that the mt4-executor
engine already polls. The two systems stay decoupled: FSP never imports the
engine; it just POSTs a command over HTTPS with the service-role key.

Everything is opt-in and fails safe:
  * disabled unless ``FSP_EXECUTE=1``;
  * ``ICT_SHADOW`` is never executed (journal-only, matching the live loop);
  * any error is logged and swallowed so a bad POST never breaks the scan loop.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from decimal import ROUND_DOWN, Decimal
from typing import Optional

import httpx

log = logging.getLogger(__name__)

# ICT_SHADOW is journal-only everywhere; never route it to execution.
_NEVER_EXECUTE = frozenset({"ICT_SHADOW"})


def size_position(
    *,
    equity: float,
    risk_pct: float,
    risk_r: float,
    inv_pips: float,
    symbol: str,
    contract_size: float = 100_000.0,
    min_lot: float = 0.01,
    max_lot: float = 0.10,
    lot_step: float = 0.01,
) -> float:
    """Convert a risk budget into a broker lot size.

    Dollar risk = ``equity * risk_pct * risk_r`` (matches FSP's existing
    convention in cli/main.py). Loss on 1.0 lot if the stop is hit is
    ``inv_pips * pip_size * contract_size`` in the quote currency -- exact when
    the account currency equals the quote currency (the USD-quoted majors:
    EURUSD, GBPUSD, AUDUSD...), approximate otherwise. The ``max_lot`` cap is
    the hard safety net for the approximate cases.

    Returns lots snapped *down* to ``lot_step`` and clamped to ``max_lot``.
    Returns ``0.0`` (skip) when inputs are unusable or the sized lot rounds
    below ``min_lot`` (too small to place without over-risking).
    """
    if equity <= 0 or risk_pct <= 0 or risk_r <= 0 or inv_pips <= 0:
        return 0.0

    pip_size = 0.01 if "JPY" in symbol.upper() else 0.0001
    loss_per_lot = inv_pips * pip_size * contract_size
    if loss_per_lot <= 0:
        return 0.0

    dollar_risk = equity * risk_pct * risk_r
    raw_lots = dollar_risk / loss_per_lot

    step = Decimal(str(lot_step))
    snapped = (Decimal(str(raw_lots)) / step).to_integral_value(rounding=ROUND_DOWN) * step
    lots = float(snapped)

    if lots > max_lot:
        lots = max_lot
    if lots < min_lot:
        return 0.0
    return lots


@dataclass
class ExecutionConfig:
    """Bridge configuration, resolved from environment for ECS/Docker."""

    enabled: bool = False
    supabase_url: str = ""
    supabase_key: str = ""
    bot_id: str = "default"
    max_lot: float = 0.10
    min_lot: float = 0.01
    lot_step: float = 0.01
    contract_size: float = 100_000.0
    # None => all strategies (except ICT_SHADOW). A set => only these.
    strategies: Optional[frozenset] = None
    timeout: float = 15.0

    @classmethod
    def from_env(cls) -> "ExecutionConfig":
        raw_strats = os.environ.get("FSP_EXECUTE_STRATEGIES", "").strip()
        strategies = (
            frozenset(s.strip().upper() for s in raw_strats.split(",") if s.strip())
            or None
        ) if raw_strats else None
        return cls(
            enabled=os.environ.get("FSP_EXECUTE", "").strip() == "1",
            supabase_url=os.environ.get("SUPABASE_URL", "").strip(),
            supabase_key=os.environ.get("SUPABASE_SERVICE_KEY", "").strip(),
            bot_id=os.environ.get("FSP_EXECUTE_BOT_ID", "default").strip() or "default",
            max_lot=float(os.environ.get("FSP_EXECUTE_MAX_LOT", "0.10")),
            min_lot=float(os.environ.get("FSP_EXECUTE_MIN_LOT", "0.01")),
            lot_step=float(os.environ.get("FSP_EXECUTE_LOT_STEP", "0.01")),
            contract_size=float(os.environ.get("FSP_EXECUTE_CONTRACT_SIZE", "100000")),
            strategies=strategies,
        )

    @property
    def ready(self) -> bool:
        """True when the bridge is both enabled and has Supabase credentials."""
        return self.enabled and bool(self.supabase_url) and bool(self.supabase_key)


def _should_execute(sig, decision: str, cfg: ExecutionConfig) -> bool:
    if not cfg.ready:
        return False
    if str(decision).upper() == "SKIP":
        return False
    if sig.strategy in _NEVER_EXECUTE:
        return False
    if cfg.strategies is not None and sig.strategy not in cfg.strategies:
        return False
    if sig.direction not in ("long", "short"):
        log.warning("execute: unknown direction %r for %s", sig.direction, sig.pair)
        return False
    return True


async def maybe_execute(
    sig,
    equity: float,
    risk_pct: float,
    decision: str = "TAKE",
    cfg: Optional[ExecutionConfig] = None,
) -> Optional[str]:
    """Enqueue a live order for ``sig`` if the bridge is enabled and the signal
    is executable. Returns a human-readable summary on success, else ``None``.

    ``decision`` is the LLM verdict: TAKE (full), REDUCE (half risk) or SKIP
    (no order). Never raises -- failures are logged and swallowed.
    """
    cfg = cfg or ExecutionConfig.from_env()
    if not _should_execute(sig, decision, cfg):
        return None

    risk_r = sig.risk_r
    if str(decision).upper() == "REDUCE":
        risk_r *= 0.5

    volume = size_position(
        equity=equity,
        risk_pct=risk_pct,
        risk_r=risk_r,
        inv_pips=sig.inv_pips,
        symbol=sig.pair,
        contract_size=cfg.contract_size,
        min_lot=cfg.min_lot,
        max_lot=cfg.max_lot,
        lot_step=cfg.lot_step,
    )
    if volume <= 0:
        log.info(
            "execute: %s %s sized to 0 lots (equity=%.2f risk_pct=%.4f risk_r=%.2f "
            "inv_pips=%.1f) -- skipping",
            sig.pair, sig.strategy, equity, risk_pct, risk_r, sig.inv_pips,
        )
        return None

    cmd_type = "buy" if sig.direction == "long" else "sell"
    payload = {
        "symbol": sig.pair,
        "volume": volume,
        "sl": sig.sl,
        "tp": sig.tp1,
        "comment": f"fsp:{sig.strategy}",
    }
    body = {"bot_id": cfg.bot_id, "type": cmd_type, "payload": payload}

    url = cfg.supabase_url.rstrip("/") + "/rest/v1/commands"
    headers = {
        # New-format Supabase keys (sb_secret_...) are NOT JWTs -- send on the
        # apikey header only; Authorization: Bearer makes the platform parse
        # them as a JWT and reject the request.
        "apikey": cfg.supabase_key,
        "Content-Type": "application/json",
        "Prefer": "return=minimal",
    }
    try:
        async with httpx.AsyncClient(timeout=cfg.timeout) as client:
            resp = await client.post(url, json=body, headers=headers)
        if resp.status_code >= 300:
            log.warning(
                "execute: command POST failed %s: %s", resp.status_code, resp.text[:300]
            )
            return None
    except Exception as exc:  # noqa: BLE001 -- never break the scan loop
        log.warning("execute: command POST error: %s", exc)
        return None

    return f"{cmd_type} {sig.pair} {volume} lots (sl={sig.sl:.5f} tp={sig.tp1:.5f})"
