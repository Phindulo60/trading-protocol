"""DynamoDB-backed trade journal — persistent across Fargate restarts.

The SQLite journal (db.py) lives on ephemeral container storage and is wiped
on every task restart. This backend persists every signal + outcome durably.

Table: fsp-journal (single-table design)
  PK  signal_id  (S)  — "{ts_iso}#{dedup_key}" — unique, time-sortable
  GSI by_pair    PK pair (S)   SK ts (S)    — recent-signal lookups per pair
  GSI by_status  PK status (S) SK ts (S)    — OPEN signals for the resolver

Activated by FSP_JOURNAL_BACKEND=dynamo. Table name override: FSP_DYNAMO_TABLE.
Numbers stored as Decimal (DynamoDB requirement); converted back to float on read.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any

import boto3
from boto3.dynamodb.conditions import Key

TABLE_NAME = os.environ.get("FSP_DYNAMO_TABLE", "fsp-journal")
REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")


def _to_decimal(obj: Any) -> Any:
    """Recursively convert floats to Decimal for DynamoDB. NaN/inf -> None."""
    if isinstance(obj, float):
        # DynamoDB rejects NaN/Infinity
        if obj != obj or obj in (float("inf"), float("-inf")):
            return None
        return Decimal(str(obj))
    if isinstance(obj, dict):
        return {k: _to_decimal(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_decimal(v) for v in obj]
    return obj


def _from_decimal(obj: Any) -> Any:
    """Recursively convert Decimal back to float/int for app use."""
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _from_decimal(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_from_decimal(v) for v in obj]
    return obj


class DynamoJournal:
    """Persistent journal backend. Mirrors the db.py module function signatures."""

    def __init__(self, table_name: str | None = None, region: str | None = None):
        # Resolve env vars at instantiation (not import) so tests/overrides work
        table_name = table_name or os.environ.get("FSP_DYNAMO_TABLE", "fsp-journal")
        region = region or os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
        self._ddb = boto3.resource("dynamodb", region_name=region)
        self.table = self._ddb.Table(table_name)

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _signal_id(ts: str, dedup_key: str) -> str:
        return f"{ts}#{dedup_key}"

    def _put(self, item: dict) -> str:
        """Write an item (floats -> Decimal). Returns signal_id."""
        clean = _to_decimal({k: v for k, v in item.items() if v is not None})
        self.table.put_item(Item=clean)
        return item["signal_id"]

    # ── writes ───────────────────────────────────────────────────────────────

    def log_signal(self, setup, dedup_key: str, sent: bool) -> str:
        """Log a 4SP grader setup."""
        ts = datetime.now(timezone.utc).isoformat()
        sid = self._signal_id(ts, dedup_key)
        item = {
            "signal_id": sid,
            "ts": ts,
            "table_type": "4sp",
            "pair": setup.pair,
            "grade": setup.grade.value,
            "direction": setup.direction,
            "entry": setup.entry,
            "sl": setup.sl,
            "tp1": setup.tp1,
            "tp2": setup.tp2,
            "risk_r": setup.risk_r,
            "checklist": [(c_.name, bool(c_.passed), c_.note) for c_ in setup.checklist],
            "context": {k: (bool(v) if isinstance(v, bool) else v)
                        for k, v in setup.context.items()},
            "sent_to_telegram": bool(sent),
            "dedup_key": dedup_key,
            "status": "OPEN",
        }
        # context may carry non-serialisable values — round-trip through json
        item["context"] = json.loads(json.dumps(item["context"], default=str))
        return self._put(item)

    def log_intraday_signal(self, sig, dedup_key: str, sent: bool) -> str:
        """Log an intraday strategy signal. Idempotent on signal_id."""
        sid = self._signal_id(sig.ts, dedup_key)
        # Skip if already present (mirrors SQLite INSERT OR IGNORE on dedup_key)
        existing = self.table.get_item(Key={"signal_id": sid}).get("Item")
        if existing:
            return sid
        item = {
            "signal_id": sid,
            "ts": sig.ts,
            "table_type": "intraday",
            "pair": sig.pair,
            "strategy": sig.strategy,
            "direction": sig.direction,
            "entry": sig.entry,
            "sl": sig.sl,
            "tp1": sig.tp1,
            "tp2": sig.tp2,
            "inv_pips": sig.inv_pips,
            "rr_tp1": sig.rr_tp1,
            "risk_r": sig.risk_r,
            "note": sig.note,
            "dedup_key": dedup_key,
            "sent_to_telegram": bool(sent),
            "context": json.loads(json.dumps(sig.context, default=str)),
            "status": "OPEN",
        }
        return self._put(item)

    # ── updates ──────────────────────────────────────────────────────────────

    def _update(self, signal_id: str, expr: str, values: dict) -> None:
        self.table.update_item(
            Key={"signal_id": signal_id},
            UpdateExpression=expr,
            ExpressionAttributeValues=_to_decimal(values),
        )

    def update_features(self, signal_id: str, features: dict) -> None:
        feats = json.loads(json.dumps(features, default=str))
        self._update(signal_id, "SET features = :f", {":f": feats})

    def update_meta_prediction(self, signal_id: str, p_win: float) -> None:
        self._update(signal_id, "SET meta_p_win = :p", {":p": float(p_win)})

    def update_outcome(self, signal_id: str, outcome: str,
                       r_multiple: float, exit_ts: str) -> None:
        # status -> RESOLVED so it drops out of the resolver's OPEN query
        self.table.update_item(
            Key={"signal_id": signal_id},
            UpdateExpression=("SET outcome = :o, r_multiple = :r, "
                              "exit_ts = :e, #s = :st"),
            ExpressionAttributeNames={"#s": "status"},
            ExpressionAttributeValues=_to_decimal({
                ":o": outcome, ":r": float(r_multiple),
                ":e": exit_ts, ":st": "RESOLVED",
            }),
        )

    # ── queries ──────────────────────────────────────────────────────────────

    def last_signal_dedup_key(self, pair: str, minutes: int = 60,
                              strategy: str | None = None) -> str | None:
        """Most recent dedup_key for pair within `minutes`. Uses by_pair GSI."""
        cutoff = (datetime.now(timezone.utc) - timedelta(minutes=minutes)).isoformat()
        resp = self.table.query(
            IndexName="by_pair",
            KeyConditionExpression=Key("pair").eq(pair) & Key("ts").gte(cutoff),
            ScanIndexForward=False,  # newest first
            Limit=25,
        )
        for item in resp.get("Items", []):
            if strategy is not None and item.get("strategy") != strategy:
                continue
            if strategy is None and item.get("table_type") != "4sp":
                # 4SP path ignores strategy; only match 4sp rows
                continue
            return item.get("dedup_key")
        return None

    def unresolved_signals(self, strategy: str | None = "TREND_RSI") -> list[dict]:
        """All OPEN signals, optionally filtered by strategy. Uses by_status GSI."""
        items: list[dict] = []
        kwargs = {
            "IndexName": "by_status",
            "KeyConditionExpression": Key("status").eq("OPEN"),
            "ScanIndexForward": True,
        }
        while True:
            resp = self.table.query(**kwargs)
            items.extend(resp.get("Items", []))
            lek = resp.get("LastEvaluatedKey")
            if not lek:
                break
            kwargs["ExclusiveStartKey"] = lek

        out = []
        for it in items:
            if it.get("table_type") != "intraday":
                continue
            if strategy is not None and it.get("strategy") != strategy:
                continue
            out.append({
                "id": it["signal_id"], "ts": it["ts"], "pair": it["pair"],
                "strategy": it.get("strategy"), "direction": it["direction"],
                "entry": _from_decimal(it.get("entry")),
                "sl": _from_decimal(it.get("sl")),
                "tp1": _from_decimal(it.get("tp1")),
                "rr_tp1": _from_decimal(it.get("rr_tp1")),
                "context": _from_decimal(it.get("context", {})),
            })
        out.sort(key=lambda r: r["ts"])
        return out

    def resolved_signals(self, strategy: str = "TREND_RSI") -> list[dict]:
        """All RESOLVED signals for a strategy. Uses by_status GSI."""
        items: list[dict] = []
        kwargs = {
            "IndexName": "by_status",
            "KeyConditionExpression": Key("status").eq("RESOLVED"),
            "ScanIndexForward": True,
        }
        while True:
            resp = self.table.query(**kwargs)
            items.extend(resp.get("Items", []))
            lek = resp.get("LastEvaluatedKey")
            if not lek:
                break
            kwargs["ExclusiveStartKey"] = lek

        out = []
        for it in items:
            if it.get("strategy") != strategy:
                continue
            out.append({
                "id": it["signal_id"], "ts": it["ts"], "pair": it["pair"],
                "direction": it["direction"],
                "entry": _from_decimal(it.get("entry")),
                "sl": _from_decimal(it.get("sl")),
                "tp1": _from_decimal(it.get("tp1")),
                "rr_tp1": _from_decimal(it.get("rr_tp1")),
                "outcome": it.get("outcome"),
                "r_multiple": _from_decimal(it.get("r_multiple")),
                "exit_ts": it.get("exit_ts"),
                "context": _from_decimal(it.get("context", {})),
            })
        out.sort(key=lambda r: r["ts"])
        return out


# ── table provisioning (idempotent) ──────────────────────────────────────────

def ensure_table(table_name: str = TABLE_NAME, region: str = REGION) -> None:
    """Create the fsp-journal table + GSIs if absent. Safe to call repeatedly."""
    ddb = boto3.client("dynamodb", region_name=region)
    existing = ddb.list_tables()["TableNames"]
    if table_name in existing:
        return
    ddb.create_table(
        TableName=table_name,
        BillingMode="PAY_PER_REQUEST",
        AttributeDefinitions=[
            {"AttributeName": "signal_id", "AttributeType": "S"},
            {"AttributeName": "pair", "AttributeType": "S"},
            {"AttributeName": "ts", "AttributeType": "S"},
            {"AttributeName": "status", "AttributeType": "S"},
        ],
        KeySchema=[{"AttributeName": "signal_id", "KeyType": "HASH"}],
        GlobalSecondaryIndexes=[
            {
                "IndexName": "by_pair",
                "KeySchema": [
                    {"AttributeName": "pair", "KeyType": "HASH"},
                    {"AttributeName": "ts", "KeyType": "RANGE"},
                ],
                "Projection": {"ProjectionType": "ALL"},
            },
            {
                "IndexName": "by_status",
                "KeySchema": [
                    {"AttributeName": "status", "KeyType": "HASH"},
                    {"AttributeName": "ts", "KeyType": "RANGE"},
                ],
                "Projection": {"ProjectionType": "ALL"},
            },
        ],
    )
    ddb.get_waiter("table_exists").wait(TableName=table_name)
