"""Config file for fsp — stored at ~/.fsp/config.toml (or FSP_DATA_DIR)."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import tomllib

from fsp.config import data_dir

CONFIG_PATH = data_dir() / "config.toml"


def parse_chat_ids(raw: str) -> tuple[str, list[str]]:
    """Parse a chat_id string into (primary, extras).

    Format: "primary" or "primary,extra1,extra2". Comma-separated;
    whitespace stripped. The first id is the primary (DM, used for
    command authorization). Subsequent ids are fan-out targets only.
    """
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        raise ValueError("chat_id is empty")
    return parts[0], parts[1:]


def load() -> dict[str, Any]:
    """Load config from file, with env var overrides for container deploys."""
    import os
    cfg: dict[str, Any] = {}
    if CONFIG_PATH.exists():
        cfg = tomllib.loads(CONFIG_PATH.read_text())

    # Env var overrides (for ECS/Docker)
    tg_token = os.environ.get("TELEGRAM_BOT_TOKEN")
    tg_chat = os.environ.get("TELEGRAM_CHAT_ID")
    if tg_token or tg_chat:
        tg = cfg.setdefault("telegram", {})
        if tg_token:
            tg["bot_token"] = tg_token
        if tg_chat:
            tg["chat_id"] = tg_chat

    return cfg


def save(data: dict[str, Any]) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for section, vals in data.items():
        lines.append(f"[{section}]")
        for k, v in vals.items():
            if isinstance(v, str):
                lines.append(f'{k} = "{v}"')
            elif isinstance(v, bool):
                lines.append(f"{k} = {'true' if v else 'false'}")
            elif isinstance(v, list):
                inner = ", ".join(f'"{x}"' for x in v)
                lines.append(f"{k} = [{inner}]")
            else:
                lines.append(f"{k} = {v}")
        lines.append("")
    CONFIG_PATH.write_text("\n".join(lines))
