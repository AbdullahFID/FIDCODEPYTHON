# logging_utils.py
# Central structured logging for Flight-Intel (Loki-ready / Loki-friendly)

import json
import logging
import os
import sys
import uuid
from datetime import datetime
from contextvars import ContextVar
from typing import Any, Dict

# Per-request correlation id (attached via middleware in api.py)
_request_id: ContextVar[str | None] = ContextVar("request_id", default=None)

SERVICE_NAME = os.getenv("SERVICE_NAME", "flightintel")
ENV = os.getenv("APP_ENV", "dev")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# Built-in LogRecord fields that must never be overwritten
_RESERVED_LOG_FIELDS = {
    "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
    "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
    "created", "msecs", "relativeCreated", "thread", "threadName",
    "processName", "process"
}


class LokiJSONFormatter(logging.Formatter):
    """
    Format log records as single-line JSON for Loki ingestion.

    Each log line produces:
        {
            "ts": "...",
            "level": "INFO",
            "logger": "flightintel.api",
            "service": "flightintel",
            "env": "dev",
            "message": "...",
            "request_id": "...",
            ... plus all structured fields ...
        }
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "ts": datetime.utcnow().isoformat(timespec="milliseconds") + "Z",
            "level": record.levelname,
            "logger": record.name,
            "service": SERVICE_NAME,
            "env": ENV,
            "message": record.getMessage(),
        }

        # Attach correlation ID if present
        rid = _request_id.get()
        if rid:
            payload["request_id"] = rid

        # Standard LogRecord keys we will *not* echo as structured fields
        standard_keys = _reserved_keys = _RESERVED_LOG_FIELDS

        # Merge user structured data safely
        for key, value in record.__dict__.items():
            if key.startswith("_"):
                continue
            if key in payload:
                continue
            if key in standard_keys:
                continue

            payload[key] = value

        return json.dumps(payload, default=str)


def configure_logging() -> None:
    """
    Configure root logging once for the whole process.
    Output = pure JSON → Loki-ready.
    """
    root = logging.getLogger()

    # Prevent double config
    if getattr(root, "_loki_configured", False):
        return

    root.setLevel(LOG_LEVEL)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(LokiJSONFormatter())

    root.handlers.clear()
    root.addHandler(handler)

    root._loki_configured = True  # type: ignore[attr-defined]


def new_request_id() -> str:
    rid = uuid.uuid4().hex
    _request_id.set(rid)
    return rid


def set_request_id(rid: str | None) -> None:
    _request_id.set(rid)


def log_event(
    logger: logging.Logger,
    event: str,
    level: int = logging.INFO,
    **fields: Any,
) -> None:
    """
    Structured logging helper.

    Ensures fields never collide with LogRecord built-ins.
    Automatically rewrites:
        filename → field_filename
        module   → field_module
        etc.
    """
    safe_fields = {}

    for key, value in fields.items():
        if key in _RESERVED_LOG_FIELDS:
            safe_fields[f"field_{key}"] = value
        else:
            safe_fields[key] = value

    logger.log(level, event, extra={"event": event, **safe_fields})
