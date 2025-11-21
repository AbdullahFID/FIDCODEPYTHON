# logging_utils.py
"""
Central logging utilities for Flight-Intel.

Provides:
    - PerfectLogger class
    - module-level `logger` instance

All other modules should:
    from logging_utils import logger
"""

import json
import logging
import time
from collections import defaultdict
from typing import Any, Dict, List

# Configure root logging once
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)


class PerfectLogger:
    def __init__(self, name: str) -> None:
        self.logger = logging.getLogger(name)
        self.timers: Dict[str, float] = {}
        self.stats: Dict[str, List[float]] = defaultdict(list)

    # ---------------- timing helpers ----------------
    def start_timer(self, operation: str) -> float:
        """
        Mark the start of a timed operation.
        Returns the timestamp so callers can store it if they want.
        """
        t = time.perf_counter()
        self.timers[operation] = t
        self.logger.info(f"⏱️  [{operation}] Started")
        return t

    def end_timer(self, operation: str) -> float:
        """
        Mark the end of a timed operation.
        Returns elapsed seconds (0.0 if no matching start).
        """
        if operation not in self.timers:
            return 0.0

        elapsed = time.perf_counter() - self.timers.pop(operation)
        self.stats[operation].append(elapsed)

        if elapsed < 1:
            emoji = "⚡"
        elif elapsed < 3:
            emoji = "✅"
        else:
            emoji = "⏰"

        self.logger.info(f"{emoji} [{operation}] Completed in {elapsed:.2f}s")
        return elapsed

    # ---------------- domain-specific helpers ----------------
    def log_extraction(self, flights_count: int, attempt: int, method: str) -> None:
        """
        Convenience helper used by the extraction engine.
        """
        if flights_count > 0:
            self.logger.info(
                f"✈️  Extracted {flights_count} flights on attempt {attempt} using {method}"
            )
        else:
            self.logger.warning(
                f"⚠️  No flights found on attempt {attempt} using {method}"
            )

    def log_api_call(
        self,
        api_name: str,
        params: Dict[str, Any],
        response: Any | None = None,
    ) -> None:
        """
        Optional helper to log upstream API calls (FlightAware, FR24, Aviation Edge, etc.).
        """
        self.logger.info(f"🌐 API CALL: {api_name}")
        try:
            self.logger.info(
                "   📤 Params: " + json.dumps(params, default=str)[:500]
            )
        except Exception:
            self.logger.info(f"   📤 Params (repr): {repr(params)[:500]}")

        if response is not None:
            try:
                self.logger.info(
                    "   📥 Response: " + json.dumps(response, default=str)[:500]
                )
            except Exception:
                self.logger.info(f"   📥 Response (repr): {repr(response)[:500]}")


# Global logger used everywhere (same as your old main.py)
logger = PerfectLogger("flight-intel")
