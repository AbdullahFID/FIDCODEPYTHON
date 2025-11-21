from __future__ import annotations

import asyncio
import re as _re
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from typing import Any, Dict, Optional, Tuple, Pattern

try:
    from zoneinfo import ZoneInfo
except ImportError:  # Py<3.9 fallback
    from backports.zoneinfo import ZoneInfo  # type: ignore

from .config import AIRPORT_TIMEZONES

# Regex helpers

FLIGHT_PATTERN: Pattern[str] = _re.compile(r"^[A-Z0-9]{2,4}\d{1,5}[A-Z]?$")
_IDENT_SPLIT_RE: Pattern[str] = _re.compile(
    r"^([A-Z]{3}|[A-Z0-9]{2})?(\d{1,5})([A-Z]?)$"
)


def _split_ident(ident: str) -> Tuple[Optional[str], str, Optional[str]]:
    """
    Returns: (airline_prefix, numeric_part, suffix)
    DL9013 -> ("DL", "9013", None)
    DAL9013A -> ("DAL", "9013", "A")
    9013 -> (None, "9013", None)
    """
    m = _IDENT_SPLIT_RE.match(ident.strip().upper())
    if not m:
        return None, ident.strip().upper(), None
    return (m.group(1), m.group(2), m.group(3) or None)


@lru_cache(maxsize=256)
def _iso_day_window_utc(year: int, month: int, day: int) -> Tuple[str, str]:
    start = datetime(year, month, day, tzinfo=timezone.utc)
    end = start + timedelta(days=1)
    return (
        start.isoformat().replace("+00:00", "Z"),
        end.isoformat().replace("+00:00", "Z"),
    )


@lru_cache(maxsize=512)
def _to_iata(code: Optional[str]) -> Optional[str]:
    if not code:
        return None
    c = code.upper().strip()
    if len(c) == 3:
        return c
    if len(c) == 4 and c.startswith("K"):  # simple KXXX→XXX (US ICAO)
        return c[1:]
    return None


def _normalize_airport_fast(obj: Any) -> Optional[str]:
    """
    Common keys seen in AeroAPI + FR24:
      - {"code_iata": "SFO"} / {"iata": "SFO"} / {"code": "SFO"}
      - Might be a plain string "SFO"
    """
    if isinstance(obj, str):
        return _to_iata(obj)
    if isinstance(obj, dict):
        for key in ("code_iata", "iata", "code"):
            if key in obj:
                return _to_iata(str(obj[key]))
    return None


_TZ_CACHE: Dict[str, ZoneInfo] = {k: ZoneInfo(v) for k, v in AIRPORT_TIMEZONES.items()}
_DEFAULT_TZ = ZoneInfo("America/New_York")


def _to_local_hhmm_from_iso(iso: str, airport_iata: Optional[str]) -> Optional[str]:
    if not iso:
        return None
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        tz = _TZ_CACHE.get(airport_iata or "", _DEFAULT_TZ)
        return dt.astimezone(tz).strftime("%H%M")
    except Exception:
        return None


def _to_local_hhmm_from_epoch_utc(
    epoch: Optional[int], airport_iata: Optional[str]
) -> Optional[str]:
    """
    FR24 fields are in epoch seconds (UTC). Convert to local HHMM using airport tz.
    """
    if not epoch:
        return None
    try:
        utc_dt = datetime.fromtimestamp(int(epoch), tz=timezone.utc)
        tz = _TZ_CACHE.get(airport_iata or "", _DEFAULT_TZ)
        return utc_dt.astimezone(tz).strftime("%H%M")
    except Exception:
        return None


class _AsyncTokenBucket:
    """Simple async token bucket limiter shared by API clients."""

    def __init__(self, rate: float, burst: int):
        self.rate = float(rate)
        self.capacity = float(burst)
        self.tokens = float(burst)
        self.t = asyncio.get_event_loop().time()
        self._lock = asyncio.Lock()

    async def acquire(self, n: float = 1.0) -> None:
        async with self._lock:
            now = asyncio.get_event_loop().time()
            # Refill
            self.tokens = min(self.capacity, self.tokens + (now - self.t) * self.rate)
            self.t = now
            if self.tokens < n:
                wait = (n - self.tokens) / self.rate
                await asyncio.sleep(wait)
                self.tokens = 0.0
            else:
                self.tokens -= n
