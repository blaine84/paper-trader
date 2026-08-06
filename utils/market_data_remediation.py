"""Remediation actions for market-data health failures."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Callable

log = logging.getLogger(__name__)


def _env_enabled(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "enabled"}


def _state_path() -> Path:
    raw = os.getenv(
        "MARKET_DATA_OUTAGE_RESTART_STATE_PATH",
        "logs/market_data_restart_state.json",
    )
    return Path(raw)


def _cooldown_seconds() -> int:
    try:
        return max(0, int(os.getenv("MARKET_DATA_OUTAGE_RESTART_COOLDOWN_SECONDS", "900")))
    except ValueError:
        return 900


def _exit_code() -> int:
    try:
        return int(os.getenv("MARKET_DATA_OUTAGE_RESTART_EXIT_CODE", "75"))
    except ValueError:
        return 75


def _allowed_source(source: str) -> bool:
    raw = os.getenv(
        "MARKET_DATA_OUTAGE_RESTART_SOURCES",
        "price_monitor,cycle_coordinator",
    )
    allowed = {item.strip() for item in raw.split(",") if item.strip()}
    return not allowed or source in allowed


def _read_last_restart(path: Path) -> float:
    try:
        return float(json.loads(path.read_text()).get("last_restart_at", 0.0))
    except Exception:
        return 0.0


def _write_restart_state(path: Path, payload: dict, now: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "last_restart_at": now,
        "payload": payload,
    }, sort_keys=True))


def maybe_restart_orchestrator_after_market_data_outage(
    payload: dict,
    *,
    restart_func: Callable[[int], None] | None = None,
    delay_seconds: float = 1.0,
) -> bool:
    """Restart the orchestrator after a critical all-symbol market-data outage.

    Returns True when a restart was scheduled. The default restart function exits
    the current process; launchd KeepAlive is expected to start it again.
    """
    if not _env_enabled("MARKET_DATA_OUTAGE_RESTART_ENABLED", "false"):
        return False

    source = str(payload.get("source") or "")
    if not _allowed_source(source):
        return False

    now = time.time()
    path = _state_path()
    last_restart_at = _read_last_restart(path)
    cooldown = _cooldown_seconds()
    if cooldown and now - last_restart_at < cooldown:
        log.error(
            "MARKET_DATA_REMEDIATION_SUPPRESSED source=%s cooldown_remaining=%.0fs",
            source,
            cooldown - (now - last_restart_at),
        )
        return False

    _write_restart_state(path, payload, now)
    code = _exit_code()
    log.critical(
        "MARKET_DATA_REMEDIATION_RESTART_SCHEDULED source=%s consumer=%s exit_code=%s",
        source,
        payload.get("consumer"),
        code,
    )

    def _restart() -> None:
        if delay_seconds > 0:
            time.sleep(delay_seconds)
        if restart_func is not None:
            restart_func(code)
        else:
            os._exit(code)

    if restart_func is not None and delay_seconds <= 0:
        _restart()
        return True

    thread = threading.Thread(
        target=_restart,
        name="market-data-remediation-restart",
        daemon=True,
    )
    thread.start()
    return True
