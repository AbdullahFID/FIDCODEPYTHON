from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import time
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple, Pattern, Union

import aiohttp
from dotenv import load_dotenv

from logging_utils import log_event

load_dotenv()

logger = logging.getLogger("flightintel.aeroapi")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG & CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

AEROAPI_BASE_URL = "https://aeroapi.flightaware.com/aeroapi"  # v4 path

# Rate limiting / concurrency knobs (per-process for AeroAPI)
AEROAPI_MAX_RPS = float(os.getenv("AEROAPI_MAX_RPS", "10"))  # tokens/sec
AEROAPI_BURST = int(os.getenv("AEROAPI_BURST", "3"))  # bucket capacity

# Lightweight response cache for identical AeroAPI GETs
_AERO_RESP_CACHE: Dict[Tuple[str, Tuple[Tuple[str, str], ...]], Tuple[float, Any]] = {}
_AERO_CACHE_TTL = 30.0  # seconds

# ─────────────────────────────────────────────────────────────────────────────
# REGEX & HELPERS (kept local for self-containment)
# ─────────────────────────────────────────────────────────────────────────────

import re as _re

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


try:
    from zoneinfo import ZoneInfo
except ImportError:  # Py<3.9
    from backports.zoneinfo import ZoneInfo  # type: ignore


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
    Common keys seen in AeroAPI:
      - {"code_iata": "SFO"} / {"iata": "SFO"} / {"code": "SFO"}
      - Or plain string "SFO"
    """
    if isinstance(obj, str):
        return _to_iata(obj)
    if isinstance(obj, dict):
        for key in ("code_iata", "iata", "code"):
            if key in obj:
                return _to_iata(str(obj[key]))
    return None


AIRPORT_TIMEZONES: Dict[str, str] = {
    # Eastern
    "JFK": "America/New_York",
    "LGA": "America/New_York",
    "EWR": "America/New_York",
    "ATL": "America/New_York",
    "BOS": "America/New_York",
    "DCA": "America/New_York",
    "MIA": "America/New_York",
    "MCO": "America/New_York",
    "TPA": "America/New_York",
    "PHL": "America/New_York",
    "CLT": "America/New_York",
    "BWI": "America/New_York",
    "DTW": "America/Detroit",
    "JAX": "America/New_York",
    # Central
    "ORD": "America/Chicago",
    "MDW": "America/Chicago",
    "DFW": "America/Chicago",
    "IAH": "America/Chicago",
    "MSP": "America/Chicago",
    "STL": "America/Chicago",
    "MCI": "America/Chicago",
    "MKE": "America/Chicago",
    "MSY": "America/Chicago",
    "BNA": "America/Chicago",
    "AUS": "America/Chicago",
    "SAT": "America/Chicago",
    # Mountain
    "DEN": "America/Denver",
    "SLC": "America/Denver",
    "PHX": "America/Phoenix",
    # Pacific
    "LAX": "America/Los_Angeles",
    "SFO": "America/Los_Angeles",
    "SEA": "America/Los_Angeles",
    "SAN": "America/Los_Angeles",
    "PDX": "America/Los_Angeles",
    "LAS": "America/Los_Angeles",
    "SJC": "America/Los_Angeles",
    "OAK": "America/Los_Angeles",
    "SMF": "America/Los_Angeles",
    # Alaska/Hawaii
    "ANC": "America/Anchorage",
    "HNL": "Pacific/Honolulu",
}
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
    if not epoch:
        return None
    try:
        utc_dt = datetime.fromtimestamp(int(epoch), tz=timezone.utc)
        tz = _TZ_CACHE.get(airport_iata or "", _DEFAULT_TZ)
        return utc_dt.astimezone(tz).strftime("%H%M")
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# RATE LIMITER
# ─────────────────────────────────────────────────────────────────────────────


class _AsyncTokenBucket:
    """Simple async token bucket limiter."""

    def __init__(self, rate: float, burst: int):
        self.rate = float(rate)
        self.capacity = float(burst)
        self.tokens = float(burst)
        self.t = asyncio.get_event_loop().time()
        self._lock = asyncio.Lock()

    async def acquire(self, n: float = 1.0):
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


# ─────────────────────────────────────────────────────────────────────────────
# AEROAPI CLIENT
# ─────────────────────────────────────────────────────────────────────────────


class AeroAPIClient:
    """
    FlightAware AeroAPI v4 client (read-only) with robust fallbacks.

    Endpoints used:
      - GET /flights/{ident}?start={iso}&end={iso}
      - GET /history/flights/{ident}?start={iso}&end={iso}
      - GET /schedules/{date_start}/{date_end}?airline={DL}&flight_number={9013}
      - GET /schedules/{date_start}/{date_end}?airline={DL}&destination={MCO}[&origin=MSP]
    """

    def __init__(self, api_key: str) -> None:
        self._headers = {"x-apikey": api_key, "Accept": "application/json"}
        self._session: Optional[aiohttp.ClientSession] = None
        self._connector = aiohttp.TCPConnector(
            limit=20,
            limit_per_host=10,
            ttl_dns_cache=300,
            enable_cleanup_closed=True,
            keepalive_timeout=30,
        )
        self._limiter = _AsyncTokenBucket(AEROAPI_MAX_RPS, AEROAPI_BURST)

    async def __aenter__(self) -> "AeroAPIClient":
        timeout = aiohttp.ClientTimeout(total=12, connect=3)
        self._session = aiohttp.ClientSession(
            headers=self._headers, connector=self._connector, timeout=timeout
        )
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._session:
            await self._session.close()

    async def _do_get(self, path: str, params: Dict[str, Any]) -> Tuple[int, Any, float]:
        """
        GET with rate limit, small cache, and 429 backoff.
        Returns (status, body, elapsed_seconds).
        """
        await self._limiter.acquire()
        url = f"{AEROAPI_BASE_URL}{path}"
        key = (path, tuple(sorted((k, str(v)) for k, v in params.items())))
        now = time.perf_counter()

        # Response de-dupe cache
        hit = _AERO_RESP_CACHE.get(key)
        if hit and (now - hit[0] < _AERO_CACHE_TTL):
            status, body = 200, hit[1]
            log_event(
                logger,
                "aeroapi_cache_hit",
                provider="flightaware",
                endpoint=path,
                cache_key=str(key),
            )
            return status, body, 0.0

        t0 = now
        attempts = 3
        status = 0
        body: Any = None

        for attempt in range(attempts):
            async with self._session.get(url, params=params) as r:
                elapsed = time.perf_counter() - t0
                status = r.status
                try:
                    body = await r.json()
                except Exception:
                    body = await r.text()

                log_event(
                    logger,
                    "aeroapi_http_call",
                    provider="flightaware",
                    endpoint=path,
                    status_code=status,
                    duration_ms=int(elapsed * 1000),
                    attempt=attempt + 1,
                )

                if status == 429 and attempt < attempts - 1:
                    ra = r.headers.get("Retry-After")
                    if ra:
                        try:
                            wait = float(ra) + 0.5
                        except Exception:
                            wait = 2.0
                    else:
                        wait = 2.0 + random.random()
                    log_event(
                        logger,
                        "aeroapi_rate_limited",
                        level=logging.WARNING,
                        provider="flightaware",
                        endpoint=path,
                        retry_after=r.headers.get("Retry-After"),
                        wait_seconds=wait,
                        attempt=attempt + 1,
                    )
                    await asyncio.sleep(wait)
                    continue

                if status == 200:
                    _AERO_RESP_CACHE[key] = (time.perf_counter(), body)
                    # Special logging for /schedules
                    if path.startswith("/schedules") and isinstance(body, dict):
                        sched = body.get("scheduled") or []
                        log_event(
                            logger,
                            "aeroapi_schedules_result",
                            count=len(sched) if isinstance(sched, list) else 0,
                        )
                    return status, body, elapsed
                return status, body, elapsed

        return status, body, time.perf_counter() - t0

    async def search(
        self,
        flight_no: str,
        date_mmddyyyy: str,
        *,
        origin_hint: Optional[str] = None,
        dest_hint: Optional[str] = None,
    ) -> Optional[Union[Dict[str, Any], List[Dict[str, Any]]]]:
        """
        Returns flight data:
        - Single dict if only one match OR all data is complete
        - List of dicts if multiple matches AND data is incomplete
        - None if no matches found
        """
        if not self._session:
            return None

        ident = (flight_no or "").strip().upper()
        if not ident or not FLIGHT_PATTERN.match(ident):
            log_event(
                logger,
                "aeroapi_invalid_ident",
                level=logging.WARNING,
                flight_no=flight_no,
                ident=ident,
            )
            return None

        # Parse date & decide endpoint
        try:
            m, d, y = [int(x) for x in date_mmddyyyy.split("/")]
            flight_dt = datetime(y, m, d, tzinfo=timezone.utc)
        except Exception:
            log_event(
                logger,
                "aeroapi_bad_date",
                level=logging.ERROR,
                date_raw=date_mmddyyyy,
                flight_no=flight_no,
            )
            return None

        day_start_iso, day_end_iso = _iso_day_window_utc(
            flight_dt.year, flight_dt.month, flight_dt.day
        )
        days_delta = (flight_dt.date() - datetime.utcnow().date()).days

        has_incomplete_data = not all([origin_hint, dest_hint])

        S_PARAMS = {
            "include_codeshares": "false",
            "max_pages": "1",
        }

        # Strategy Phase A — /flights /history /schedules
        phaseA: List[Tuple[str, Dict[str, Any]]] = []

        if days_delta < -10:
            phaseA.append(
                (
                    f"/history/flights/{ident}",
                    {"start": day_start_iso, "end": day_end_iso},
                )
            )
        elif -10 <= days_delta <= 2:
            phaseA.append(
                (f"/flights/{ident}", {"start": day_start_iso, "end": day_end_iso})
            )
        else:
            airline, number, _suffix = _split_ident(ident)
            if number:
                params = {"flight_number": number, **S_PARAMS}
                if airline:
                    params["airline"] = airline
                phaseA.append(
                    (
                        f"/schedules/{flight_dt.strftime('%Y-%m-%d')}/{(flight_dt + timedelta(days=1)).strftime('%Y-%m-%d')}",
                        params,
                    )
                )

        def _have(prefix: str, arr: List[Tuple[str, Dict[str, Any]]]) -> bool:
            return any(p.startswith(prefix) for p, _ in arr)

        if not _have("/flights/", phaseA):
            phaseA.append(
                (f"/flights/{ident}", {"start": day_start_iso, "end": day_end_iso})
            )
        if not _have("/history/", phaseA):
            phaseA.append(
                (
                    f"/history/flights/{ident}",
                    {"start": day_start_iso, "end": day_end_iso},
                )
            )

        last_error_text: Optional[str] = None

        def _extract_all_items(body: Any) -> List[Dict[str, Any]]:
            if not isinstance(body, dict):
                return []
            flights_list: List[Dict[str, Any]] = []

            if "scheduled" in body and isinstance(body["scheduled"], list):
                flights_list = body["scheduled"]
            elif "flights" in body and isinstance(body["flights"], list):
                flights_list = body["flights"]
            elif "data" in body and isinstance(body["data"], list):
                flights_list = body["data"]
            else:
                for key, val in body.items():
                    if isinstance(val, list) and val and isinstance(val[0], dict):
                        flights_list = val
                        break

            if not flights_list:
                return []

            def _matches_ident(f: Dict[str, Any]) -> bool:
                for k in ("ident", "ident_iata", "ident_icao", "flight_number"):
                    v = f.get(k)
                    if isinstance(v, str):
                        v_clean = v.strip().upper()
                        if v_clean == ident:
                            return True
                        if v_clean.endswith(ident[2:]) and len(v_clean) > len(ident):
                            return True

                airline, number, _sfx = _split_ident(ident)
                if number:
                    fn = str(f.get("flight_number") or "").strip()
                    if fn == number:
                        return True

                return False

            return [item for item in flights_list if _matches_ident(item)]

        def _extract_first_item(body: Any) -> Optional[Dict[str, Any]]:
            if not isinstance(body, dict):
                return None
            flights_list: List[Dict[str, Any]] = []

            if "scheduled" in body and isinstance(body["scheduled"], list):
                flights_list = body["scheduled"]
            elif "flights" in body and isinstance(body["flights"], list):
                flights_list = body["flights"]
            elif "data" in body and isinstance(body["data"], list):
                flights_list = body["data"]
            else:
                for key, val in body.items():
                    if isinstance(val, list) and val and isinstance(val[0], dict):
                        flights_list = val
                        break

            if not flights_list:
                return None

            def _matches_ident(f: Dict[str, Any]) -> bool:
                for k in ("ident", "ident_iata", "ident_icao", "flight_number"):
                    v = f.get(k)
                    if isinstance(v, str):
                        v_clean = v.strip().upper()
                        if v_clean == ident:
                            return True
                        if v_clean.endswith(ident[2:]) and len(v_clean) > len(ident):
                            return True

                airline, number, _sfx = _split_ident(ident)
                if number:
                    fn = str(f.get("flight_number") or "").strip()
                    if fn == number:
                        return True

                return False

            for item in flights_list:
                if _matches_ident(item):
                    return item
            return flights_list[0]

        # Phase A
        for path, params in phaseA:
            status, body, _elapsed = await self._do_get(path, params)

            if status == 400:
                err_txt = body if isinstance(body, str) else json.dumps(body)[:200]
                log_event(
                    logger,
                    "aeroapi_400",
                    level=logging.ERROR,
                    endpoint=path,
                    params_preview=str(params),
                    error_preview=err_txt,
                )
                last_error_text = err_txt
                continue
            if status in (404, 204):
                continue
            if status != 200:
                last_error_text = (
                    body if isinstance(body, str) else json.dumps(body)[:200]
                )
                continue

            if has_incomplete_data:
                items = _extract_all_items(body)
                if len(items) > 1:
                    log_event(
                        logger,
                        "aeroapi_multi_match",
                        ident=ident,
                        date=date_mmddyyyy,
                        count=len(items),
                    )
                    return items
                elif len(items) == 1:
                    return items[0]
            else:
                item = _extract_first_item(body)
                if item:
                    return item

        # Phase B — route-based schedules
        sched_path = f"/schedules/{flight_dt.strftime('%Y-%m-%d')}/{(flight_dt + timedelta(days=1)).strftime('%Y-%m-%d')}"
        airline, number, _suffix = _split_ident(ident)
        route_tries: List[Dict[str, Any]] = []
        if dest_hint:
            p = {"airline": airline or "DL", "destination": dest_hint, **S_PARAMS}
            if origin_hint:
                p["origin"] = origin_hint
            route_tries.append(p)
        if origin_hint and not dest_hint:
            route_tries.append(
                {"airline": airline or "DL", "origin": origin_hint, **S_PARAMS}
            )

        for params in route_tries:
            status, body, _elapsed = await self._do_get(sched_path, params)

            if status == 400:
                err_txt = body if isinstance(body, str) else json.dumps(body)[:200]
                log_event(
                    logger,
                    "aeroapi_400",
                    level=logging.ERROR,
                    endpoint=sched_path,
                    params_preview=str(params),
                    error_preview=err_txt,
                )
                last_error_text = err_txt
                continue
            if status in (404, 204):
                continue
            if status != 200:
                last_error_text = (
                    body if isinstance(body, str) else json.dumps(body)[:200]
                )
                continue

            if has_incomplete_data:
                items = _extract_all_items(body)
                if len(items) > 1:
                    log_event(
                        logger,
                        "aeroapi_multi_match_route",
                        ident=ident,
                        date=date_mmddyyyy,
                        count=len(items),
                    )
                    return items
                elif len(items) == 1:
                    return items[0]
            else:
                item = _extract_first_item(body)
                if item:
                    return item

        if last_error_text:
            log_event(
                logger,
                "aeroapi_no_data",
                level=logging.ERROR,
                ident=ident,
                date=date_mmddyyyy,
                last_error=last_error_text,
            )
        return None
