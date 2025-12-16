# logging_utils.py
# Central structured logging for Flight-Intel (Loki-ready / Loki-friendly)

import json
import logging
import os
import sys
import uuid
import time
from datetime import datetime
from typing import Any, Dict, Optional
from contextvars import ContextVar

# Per-request correlation id (attached via middleware in api.py)
_request_id: ContextVar[Optional[str]] = ContextVar("request_id", default=None)

SERVICE_NAME = os.getenv("SERVICE_NAME", "flightintel")
ENV = os.getenv("APP_ENV", "dev")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# Default log file path for Loki/Promtail
LOG_FILE = os.getenv("LOG_FILE", "/var/log/flightintel/app.log")

# Built-in LogRecord fields that must never be overwritten
_RESERVED_LOG_FIELDS = {
    "name",
    "msg",
    "args",
    "levelname",
    "levelno",
    "pathname",
    "filename",
    "module",
    "exc_info",
    "exc_text",
    "stack_info",
    "lineno",
    "funcName",
    "created",
    "msecs",
    "relativeCreated",
    "thread",
    "threadName",
    "processName",
    "process",
}


class LokiJSONFormatter(logging.Formatter):
    """
    Format log records as single-line JSON for Loki ingestion.

    Each log line looks like:
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

        standard_keys = _RESERVED_LOG_FIELDS

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
    Output -> JSON to both stdout and /var/log/flightintel/app.log.
    """
    root = logging.getLogger()

    # Prevent double config
    if getattr(root, "_loki_configured", False):
        return

    root.setLevel(LOG_LEVEL)

    formatter = LokiJSONFormatter()

    # 1) STDOUT (uvicorn/console)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)

    # 2) FILE for Promtail/Loki
    try:
        log_dir = os.path.dirname(LOG_FILE)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        file_handler = logging.FileHandler(LOG_FILE)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
    except Exception as e:
        # If file logging fails, keep stdout logging only
        root.error(f"Failed to set up file logging: {e}")

    root._loki_configured = True  # type: ignore[attr-defined]


def new_request_id() -> str:
    rid = uuid.uuid4().hex
    _request_id.set(rid)
    return rid


def set_request_id(rid: Optional[str]) -> None:
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
    safe_fields: Dict[str, Any] = {}

    for key, value in fields.items():
        if key in _RESERVED_LOG_FIELDS:
            safe_fields[f"field_{key}"] = value
        else:
            safe_fields[key] = value

    logger.log(level, event, extra={"event": event, **safe_fields})


class FlightIntelLogger:
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.timers: Dict[str, float] = {}

    def info(self, msg, *args, **kwargs):
        self.logger.info(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        self.logger.error(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        self.logger.warning(msg, *args, **kwargs)

    def start_timer(self, name: str):
        rid = _request_id.get() or "global"
        key = f"{rid}:{name}"
        self.timers[key] = time.time()

    def end_timer(self, name: str) -> float:
        rid = _request_id.get() or "global"
        key = f"{rid}:{name}"
        start = self.timers.pop(key, None)
        if start:
            elapsed = time.time() - start
            return elapsed
        return 0.0

    def log_extraction(self, count, attempt, method):
        self.logger.info(f"Extraction: {count} flights, attempt {attempt}, method {method}")


def get_logger(name: str) -> FlightIntelLogger:
    return FlightIntelLogger(name)
