# flight_intel_patch.py
# Flight Intel Validation & Enrichment Module — v2.3 (resilience upgrade)
#
# Changes vs v2.2:
# - Global async token-bucket rate limiter for AeroAPI + jittered 429 retry
# - Small response cache to de-duplicate identical AeroAPI requests (short TTL)
# - Tunable validator concurrency via VALIDATOR_CONCURRENCY env (default 3)
# - Smarter schedules strategy: try airline+flight_number first; only then
#   route-based (airline+destination[/origin]) to reduce calls and 429s
# - Pass optional origin/dest hints from validator → AeroAPI client
#
# Public entrypoint:
#   async def validate_extraction_results(extraction_result: Dict) -> Dict
#
# Input shape (from your extractor):
#   { "flights": [{ "date": "MM/DD/YYYY", "flight_no": "DL9013", ... }], ... }
#
# Output additions:
#   - "validation": summary dict
#   - "enriched_flights": array of EnrichedFlight (original flight + filled fields + ValidationResult)

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple, Pattern
import random
from dotenv import load_dotenv

load_dotenv()

try:
    from zoneinfo import ZoneInfo
except ImportError:  # Py<3.9 fallback
    from backports.zoneinfo import ZoneInfo  # type: ignore

import aiohttp
from pydantic import BaseModel, Field
import re as _re

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("flight-validator")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

FLIGHTAWARE_API_KEY: Optional[str] = os.getenv("FLIGHTAWARE_API_KEY")
FLIGHTRADAR24_API_KEY: Optional[str] = os.getenv("FLIGHTRADAR24_API_KEY")

AEROAPI_BASE_URL = "https://aeroapi.flightaware.com/aeroapi"  # v4 path
FR24_BASE_URL = "https://api.flightradar24.com/common/v1"  # FR24 public app API

# Rate limiting / concurrency knobs
AEROAPI_MAX_RPS = float(os.getenv("AEROAPI_MAX_RPS", "10"))  # steady tokens per second
AEROAPI_BURST = int(os.getenv("AEROAPI_BURST", "3"))  # bucket capacity
VALIDATOR_CONCURRENCY = int(os.getenv("VALIDATOR_CONCURRENCY", "1"))
AVIATION_EDGE_API_KEY: Optional[str] = os.getenv("AVIATION_EDGE_KEY")

# Cache: (flight_no_upper, date_mmddyyyy) -> (ts, ValidationResult)
_VALIDATION_CACHE: Dict[Tuple[str, str], Tuple[float, "ValidationResult"]] = {}
_CACHE_TTL_S = 15 * 60
_CACHE_MAX = 1000

# Lightweight response cache for identical AeroAPI GETs
_AERO_RESP_CACHE: Dict[Tuple[str, Tuple[Tuple[str, str], ...]], Tuple[float, Any]] = {}
_AERO_CACHE_TTL = 30.0  # seconds

# ─────────────────────────────────────────────────────────────────────────────
# REGEX & SMALL HELPERS
# ─────────────────────────────────────────────────────────────────────────────

# Examples: DL9013, DAL9013, 9E1234, UA1, AA12345, AF1234A
FLIGHT_PATTERN: Pattern[str] = _re.compile(r"^[A-Z0-9]{2,4}\d{1,5}[A-Z]?$")
_IDENT_SPLIT_RE: Pattern[str] = _re.compile(
    r"^([A-Z]{3}|[A-Z0-9]{2})?(\d{1,5})([A-Z]?)$"
)

# Right after the import section, add:
print("=" * 60)
print("🔑 API KEY DEBUG:")
print(
    f"   FLIGHTAWARE_API_KEY: {'SET' if FLIGHTAWARE_API_KEY else 'MISSING'} (len={len(FLIGHTAWARE_API_KEY or '')})"
)
print(
    f"   FLIGHTRADAR24_API_KEY: {'SET' if FLIGHTRADAR24_API_KEY else 'MISSING'} (len={len(FLIGHTRADAR24_API_KEY or '')})"
)
print(
    f"   AVIATION_EDGE_API_KEY: {'SET' if AVIATION_EDGE_API_KEY else 'MISSING'} (len={len(AVIATION_EDGE_API_KEY or '')})"
)
print("=" * 60)


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


# Minimal, high-hit map of US airports → tz
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


# ─────────────────────────────────────────────────────────────────────────────
# MODELS
# ─────────────────────────────────────────────────────────────────────────────


class ValidationResult(BaseModel):
    is_valid: bool
    confidence: float
    source: str
    warnings: List[str] = Field(default_factory=list)
    corrections: Dict[str, Any] = Field(default_factory=dict)
    filled_fields: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        validate_assignment = False
        use_enum_values = True


class EnrichedFlight(BaseModel):
    date: str
    flight_no: str
    origin: Optional[str] = None
    dest: Optional[str] = None
    sched_out_local: Optional[str] = None
    sched_in_local: Optional[str] = None
    validation_result: Optional[ValidationResult] = None

    class Config:
        validate_assignment = False
        use_enum_values = True


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
# AEROAPI CLIENT (FlightAware)
# ─────────────────────────────────────────────────────────────────────────────


class AeroAPIClient:
    """
    FlightAware AeroAPI v4 client (read-only) with robust fallbacks.

    Endpoints used:
      - GET /flights/{ident}?start={iso}&end={iso}
      - GET /history/flights/{ident}?start={iso}&end={iso}
      - GET /schedules/{date_start}/{date_end}?airline={DL}&flight_number={9013}
      - GET /schedules/{date_start}/{date_end}?airline={DL}&destination={MCO}[&origin=MSP]

    Notes:
      • For flights beyond the 2-day live window, /schedules is preferred.
      • We try flight_number first; only if empty do we try route filters.
      • All requests pass query params via aiohttp and are rate-limited.
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
            logger.warning(f"AeroAPI: invalid flight ident '{ident}'")
            return None

        # Parse date & decide endpoint
        try:
            m, d, y = [int(x) for x in date_mmddyyyy.split("/")]
            flight_dt = datetime(y, m, d, tzinfo=timezone.utc)
        except Exception:
            logger.error(f"AeroAPI: bad date '{date_mmddyyyy}'")
            return None

        day_start_iso, day_end_iso = _iso_day_window_utc(
            flight_dt.year, flight_dt.month, flight_dt.day
        )
        days_delta = (flight_dt.date() - datetime.utcnow().date()).days

        # 🔥 Determine if we have incomplete data
        has_incomplete_data = not all([origin_hint, dest_hint])  # Missing origin or dest
        
        # Shared schedules params
        S_PARAMS = {
            "include_codeshares": "false",
            "max_pages": "1",
        }

        async def _do_get(path: str, params: Dict[str, Any]) -> Tuple[int, Any, float]:
            """GET with rate limit, small cache, and 429 backoff."""
            await self._limiter.acquire()
            url = f"{AEROAPI_BASE_URL}{path}"
            key = (path, tuple(sorted((k, str(v)) for k, v in params.items())))
            now = time.perf_counter()

            # Response de-dupe cache
            hit = _AERO_RESP_CACHE.get(key)
            if hit and (now - hit[0] < _AERO_CACHE_TTL):
                status, body = 200, hit[1]
                logger.info(
                    f"AeroAPI GET (cache) {path} params={params} status={status}"
                )
                return status, body, 0.0

            t0 = now
            attempts = 3
            for attempt in range(attempts):
                async with self._session.get(url, params=params) as r:
                    elapsed = time.perf_counter() - t0
                    status = r.status
                    try:
                        body = await r.json()
                    except Exception:
                        body = await r.text()
                    logger.info(
                        f"AeroAPI GET {path} params={params} status={status} took={elapsed:.2f}s"
                    )

                    if status == 429 and attempt < attempts - 1:
                        ra = r.headers.get("Retry-After")
                        if ra:
                            try:
                                await asyncio.sleep(
                                    float(ra) + 0.5
                                )  # Add extra 0.5s buffer
                            except Exception:
                                await asyncio.sleep(2.0)  # Increase from 0.8
                        else:
                            await asyncio.sleep(
                                2.0 + random.random()
                            )  # Increase base delay
                        continue

                    if status == 200:
                        _AERO_RESP_CACHE[key] = (time.perf_counter(), body)
                        if path.startswith("/schedules") and isinstance(body, dict):
                            if "scheduled" in body:
                                logger.info(
                                    f"   Found {len(body['scheduled'])} scheduled flights"
                                )
                                if body["scheduled"]:
                                    first = body["scheduled"][0]
                                    logger.info(
                                        f"   First flight: {first.get('ident_iata')} from {first.get('origin_iata')} to {first.get('destination_iata')}"
                                    )
                        return status, body, elapsed
                    return status, body, elapsed
            return status, body, elapsed

        # Strategy Phase A — live/near/old vs future
        phaseA: List[Tuple[str, Dict[str, Any]]] = []

        if days_delta < -10:
            # Older than ~10 days → /history
            phaseA.append(
                (
                    f"/history/flights/{ident}",
                    {"start": day_start_iso, "end": day_end_iso},
                )
            )
        elif -10 <= days_delta <= 2:
            # within 2 days → /flights
            phaseA.append(
                (f"/flights/{ident}", {"start": day_start_iso, "end": day_end_iso})
            )
        else:
            # Future/schedules window → airline+flight_number first
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

        # Always allow cross-try of flights/history to be safe (won't match far future)
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

        # 🔥 NEW: Helper to extract ALL matching items
        def _extract_all_items(body: Any) -> List[Dict[str, Any]]:
            """Extract ALL flights that match the ident."""
            if not isinstance(body, dict):
                return []
            
            flights_list: List[Dict[str, Any]] = []

            # Get the flights array from various response formats
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

            # Filter to only matching flights
            def _matches_ident(f: Dict[str, Any]) -> bool:
                # Check all possible ident fields
                for k in ("ident", "ident_iata", "ident_icao", "flight_number"):
                    v = f.get(k)
                    if isinstance(v, str):
                        v_clean = v.strip().upper()
                        # Direct match
                        if v_clean == ident:
                            return True
                        # Match without airline prefix (e.g., "DAL9013" matches "DL9013")
                        if v_clean.endswith(ident[2:]) and len(v_clean) > len(ident):
                            return True

                # Also check by flight number alone
                airline, number, _sfx = _split_ident(ident)
                if number:
                    fn = str(f.get("flight_number") or "").strip()
                    if fn == number:
                        return True

                return False

            matching = [item for item in flights_list if _matches_ident(item)]
            return matching

        # Helper to parse and pick first item (existing logic)
        def _extract_first_item(body: Any) -> Optional[Dict[str, Any]]:
            if not isinstance(body, dict):
                return None
            flights_list: List[Dict[str, Any]] = []

            # Add "scheduled" to the list of fields to check (this is what /schedules returns)
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

            # Prefer exact ident match; else first
            def _matches_ident(f: Dict[str, Any]) -> bool:
                # Check all possible ident fields
                for k in ("ident", "ident_iata", "ident_icao", "flight_number"):
                    v = f.get(k)
                    if isinstance(v, str):
                        v_clean = v.strip().upper()
                        # Direct match
                        if v_clean == ident:
                            return True
                        # Match without airline prefix (e.g., "DAL9013" matches "DL9013")
                        if v_clean.endswith(ident[2:]) and len(v_clean) > len(ident):
                            return True

                # Also check by flight number alone
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

        # Phase A tries
        for path, params in phaseA:
            status, body, _elapsed = await _do_get(path, params)

            if status == 400:
                err_txt = body if isinstance(body, str) else json.dumps(body)[:200]
                logger.error(f"AeroAPI 400 on {path} params={params} err={err_txt}")
                last_error_text = err_txt
                continue
            if status in (404, 204):
                continue
            if status != 200:
                last_error_text = (
                    body if isinstance(body, str) else json.dumps(body)[:200]
                )
                continue

            # 🔥 NEW: If data incomplete, try to get all matches
            if has_incomplete_data:
                items = _extract_all_items(body)
                if len(items) > 1:
                    logger.info(f"   📋 Found {len(items)} matching flights - returning all due to incomplete data")
                    return items
                elif len(items) == 1:
                    return items[0]
            else:
                # Complete data - just get best match
                item = _extract_first_item(body)
                if item:
                    return item

        # Phase B — only if Phase A yielded nothing: try route-based schedules
        # Use hints if available from extractor (dest first; include origin if present)
        sched_path = f"/schedules/{flight_dt.strftime('%Y-%m-%d')}/{(flight_dt + timedelta(days=1)).strftime('%Y-%m-%d')}"
        airline, number, _suffix = _split_ident(ident)
        route_tries: List[Dict[str, Any]] = []
        if dest_hint:
            p = {"airline": airline or "DL", "destination": dest_hint, **S_PARAMS}
            if origin_hint:
                p["origin"] = origin_hint
            route_tries.append(p)
        # If no dest hint, try origin-only (less precise but sometimes returns one)
        if origin_hint and not dest_hint:
            route_tries.append(
                {"airline": airline or "DL", "origin": origin_hint, **S_PARAMS}
            )

        for params in route_tries:
            status, body, _elapsed = await _do_get(sched_path, params)

            if status == 400:
                err_txt = body if isinstance(body, str) else json.dumps(body)[:200]
                logger.error(
                    f"AeroAPI 400 on {sched_path} params={params} err={err_txt}"
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

            # 🔥 NEW: Same logic for Phase B
            if has_incomplete_data:
                items = _extract_all_items(body)
                if len(items) > 1:
                    logger.info(f"   📋 Found {len(items)} matching flights - returning all due to incomplete data")
                    return items
                elif len(items) == 1:
                    return items[0]
            else:
                item = _extract_first_item(body)
                if item:
                    return item

        if last_error_text:
            logger.error(f"AeroAPI no data; last error: {last_error_text}")
        return None

# ─────────────────────────────────────────────────────────────────────────────
# FLIGHTRADAR24 CLIENT (best-effort)
# ─────────────────────────────────────────────────────────────────────────────


# flight_intel_patch.py - REPLACE the FR24Client class

class FR24Client:
    """
    FlightRadar24 API client with enterprise-grade reliability matching AeroAPIClient.
    
    Rate limit: 30 requests/minute (Essential Plan)
    Features: Token bucket limiter, response caching, 429 retry, multiple result handling
    
    Endpoint used:
      - GET /flight/list.json?query={IDENT}&fetchBy=flight&limit=25&token={API_KEY}
    """

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key
        self._session: Optional[aiohttp.ClientSession] = None
        self._connector = aiohttp.TCPConnector(
            limit=20,
            limit_per_host=10,
            ttl_dns_cache=300,
            enable_cleanup_closed=True,
            keepalive_timeout=30,
        )
        # Rate limiting: 30/min = 0.5/sec with burst capacity of 5
        self._limiter = _AsyncTokenBucket(rate=0.5, burst=5)
        
        # Response cache: (ident, date) -> (timestamp, data)
        self._cache: Dict[Tuple[str, str], Tuple[float, Any]] = {}
        self._cache_ttl = 30.0  # seconds
        self._cache_max_size = 200

    async def __aenter__(self) -> "FR24Client":
        timeout = aiohttp.ClientTimeout(total=15, connect=4)
        self._session = aiohttp.ClientSession(
            headers={
                "Accept": "application/json",
                "User-Agent": "Flight-Intel/2.3",
            },
            connector=self._connector,
            timeout=timeout,
        )
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._session:
            await self._session.close()

    def _get_from_cache(self, ident: str, date: str) -> Optional[Any]:
        """Check response cache for recent identical request."""
        key = (ident.upper(), date)
        if key in self._cache:
            ts, data = self._cache[key]
            if time.perf_counter() - ts < self._cache_ttl:
                logger.info(f"FR24 cache HIT for {ident} on {date}")
                return data
            else:
                # Expired entry
                del self._cache[key]
        return None

    def _put_in_cache(self, ident: str, date: str, data: Any) -> None:
        """Store successful response in cache."""
        key = (ident.upper(), date)
        self._cache[key] = (time.perf_counter(), data)
        
        # Evict oldest entries if cache is full
        if len(self._cache) > self._cache_max_size:
            oldest_key = min(self._cache.items(), key=lambda x: x[1][0])[0]
            del self._cache[oldest_key]

    async def _do_get(
        self, url: str, params: Dict[str, Any], attempt: int = 1
    ) -> Tuple[int, Any, float]:
        """
        Execute GET request with rate limiting, caching, and retry logic.
        Returns: (status_code, response_data, elapsed_time)
        """
        # Apply rate limiting
        await self._limiter.acquire()
        
        t0 = time.perf_counter()
        max_attempts = 3
        
        for retry in range(max_attempts):
            try:
                async with self._session.get(url, params=params) as r:
                    elapsed = time.perf_counter() - t0
                    status = r.status
                    
                    # Parse response
                    try:
                        body = await r.json()
                    except Exception:
                        body = await r.text()
                    
                    logger.info(
                        f"FR24 GET {url.split('/')[-1]} "
                        f"status={status} took={elapsed:.2f}s attempt={retry+1}/{max_attempts}"
                    )
                    
                    # Handle 429 rate limit
                    if status == 429 and retry < max_attempts - 1:
                        retry_after = r.headers.get("Retry-After")
                        if retry_after:
                            try:
                                wait_time = float(retry_after) + 1.0  # Add buffer
                            except ValueError:
                                wait_time = 3.0
                        else:
                            # Exponential backoff with jitter
                            wait_time = (2 ** retry) + random.uniform(0.5, 2.0)
                        
                        logger.warning(
                            f"FR24 429 rate limit hit - waiting {wait_time:.1f}s before retry"
                        )
                        await asyncio.sleep(wait_time)
                        continue
                    
                    # Handle other errors
                    if status != 200:
                        error_text = body if isinstance(body, str) else json.dumps(body)[:200]
                        logger.error(f"FR24 error {status}: {error_text}")
                    
                    return status, body, elapsed
                    
            except asyncio.TimeoutError:
                logger.error(f"FR24 timeout on attempt {retry+1}/{max_attempts}")
                if retry < max_attempts - 1:
                    await asyncio.sleep(1.0 + random.uniform(0, 0.5))
                    continue
                return 504, None, time.perf_counter() - t0
                
            except Exception as e:
                logger.error(f"FR24 exception on attempt {retry+1}/{max_attempts}: {e}")
                if retry < max_attempts - 1:
                    await asyncio.sleep(1.0)
                    continue
                return 500, None, time.perf_counter() - t0
        
        # All retries exhausted
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
        Search for flight data on FR24.
        
        Returns:
        - Single dict if only one match OR data is complete
        - List of dicts if multiple matches AND data is incomplete
        - None if no matches found
        
        Args:
            flight_no: Flight number (e.g., "DL9013")
            date_mmddyyyy: Date string "MM/DD/YYYY"
            origin_hint: Optional origin airport code for validation
            dest_hint: Optional destination airport code for validation
        """
        if not self._session or not self._api_key:
            logger.warning("FR24 client not initialized or missing API key")
            return None

        ident = flight_no.strip().upper()
        
        # Check cache first
        cached = self._get_from_cache(ident, date_mmddyyyy)
        if cached is not None:
            return cached
        
        # Build request params
        params = {
            "query": ident,
            "fetchBy": "flight",
            "limit": 25,  # Increased to catch more potential matches
            "token": self._api_key,
        }
        
        url = f"{FR24_BASE_URL}/flight/list.json"
        
        # Execute request with retry logic
        status, body, elapsed = await self._do_get(url, params)
        
        if status != 200 or not body:
            return None
        
        # Parse response structure
        try:
            flights = body.get("result", {}).get("response", {}).get("data", [])
            if not flights:
                logger.info(f"FR24 no flights found for {ident}")
                return None
            
            logger.info(f"FR24 found {len(flights)} potential matches for {ident}")
            
            # Filter to exact matches only
            def _matches_ident(flight_data: Dict[str, Any]) -> bool:
                """Check if flight matches our search ident."""
                # FR24 structure: flight_data contains various identifying fields
                for key in ("identification", "flight", "airline"):
                    if key in flight_data:
                        obj = flight_data[key]
                        if isinstance(obj, dict):
                            # Check number field
                            number = obj.get("number", {})
                            if isinstance(number, dict):
                                default_num = number.get("default", "").upper()
                                if default_num == ident:
                                    return True
                            # Check callsign
                            callsign = obj.get("callsign", "").upper()
                            if callsign == ident:
                                return True
                
                return False
            
            matching_flights = [f for f in flights if _matches_ident(f)]
            
            if not matching_flights:
                # No exact matches, return first result as best guess
                logger.warning(f"FR24 no exact match for {ident}, using first result")
                result = flights[0]
                self._put_in_cache(ident, date_mmddyyyy, result)
                return result
            
            # Determine if we have incomplete data
            has_incomplete_data = not all([origin_hint, dest_hint])
            
            # If data is incomplete and we have multiple matches, return all
            if has_incomplete_data and len(matching_flights) > 1:
                logger.info(
                    f"FR24 returning {len(matching_flights)} options for {ident} "
                    f"(incomplete data: origin={bool(origin_hint)}, dest={bool(dest_hint)})"
                )
                self._put_in_cache(ident, date_mmddyyyy, matching_flights)
                return matching_flights
            
            # Otherwise return best single match
            result = matching_flights[0]
            logger.info(f"FR24 returning single best match for {ident}")
            self._put_in_cache(ident, date_mmddyyyy, result)
            return result
            
        except Exception as e:
            logger.error(f"FR24 response parsing error: {e}")
            return None

# ─────────────────────────────────────────────────────────────────────────────
# AVIATION EDGE CLIENT (Cargo-focused)
# ─────────────────────────────────────────────────────────────────────────────

AVIATION_EDGE_BASE = "https://aviation-edge.com/v2/public"

# Cargo carriers that should prefer Aviation Edge
CARGO_CARRIERS = {
    "5X",
    "UPS",  # UPS
    "FX",
    "FDX",  # FedEx
    "5Y",
    "GTI",  # Atlas Air
    "K4",
    "CKS",  # Kalitta Air
    "NC",
    "NAC",  # Northern Air Cargo
    "GB",
    "ABX",  # ABX Air
    "3S",
    "PAC",  # Polar Air Cargo
    "M6",
    "AJT",  # Amerijet
    "CV",
    "CLX",  # Cargolux
    "KZ",
    "NCA",  # Nippon Cargo
    "PO",  # Polar Air Cargo (alternate code)
}


class AviationEdgeClient:
    """
    Aviation Edge API client optimized for cargo operations.

    Endpoints used:
      - GET /timetable?iataCode={XXX}&type=departure - Real-time schedules
      - GET /flightsFuture?iataCode={XXX}&date={YYYY-MM-DD} - Future schedules
      - GET /routes?departureIata={XXX}&arrivalIata={YYY} - Route validation
      - GET /airplaneDatabase?numberRegistration={N12345} - Aircraft lookup
    """

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key
        self._session: Optional[aiohttp.ClientSession] = None
        self._connector = aiohttp.TCPConnector(
            limit=15,
            limit_per_host=8,
            ttl_dns_cache=300,
            enable_cleanup_closed=True,
            keepalive_timeout=30,
        )

    async def __aenter__(self) -> "AviationEdgeClient":
        timeout = aiohttp.ClientTimeout(total=15, connect=4)
        self._session = aiohttp.ClientSession(
            headers={"Accept": "application/json"},
            connector=self._connector,
            timeout=timeout,
        )
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._session:
            await self._session.close()

    async def _get(self, endpoint: str, params: Dict[str, Any]) -> Tuple[int, Any]:
        """Shared GET with key injection."""
        if not self._session:
            return 503, None

        params["key"] = self._api_key
        url = f"{AVIATION_EDGE_BASE}/{endpoint}"

        try:
            async with self._session.get(url, params=params) as r:
                status = r.status
                if status == 200:
                    data = await r.json()
                    logger.info(
                        f"Aviation Edge GET /{endpoint} status={status} results={len(data) if isinstance(data, list) else 'N/A'}"
                    )
                    return status, data
                else:
                    text = await r.text()
                    logger.error(
                        f"Aviation Edge GET /{endpoint} status={status} error={text[:200]}"
                    )
                    return status, None
        except asyncio.TimeoutError:
            logger.error(f"Aviation Edge GET /{endpoint} TIMEOUT")
            return 504, None
        except Exception as e:
            logger.error(f"Aviation Edge GET /{endpoint} exception: {e}")
            return 500, None

    async def search(
        self,
        flight_no: str,
        date_mmddyyyy: str,
        *,
        origin_hint: Optional[str] = None,
        dest_hint: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Search for cargo flight using multiple strategies.

        Priority:
        1. /timetable (for today/tomorrow)
        2. /flightsFuture (for dates >2 days out)
        3. /routes (fallback for route validation)
        """
        ident = flight_no.strip().upper()

        # Parse date
        try:
            m, d, y = [int(x) for x in date_mmddyyyy.split("/")]
            flight_date = datetime(y, m, d, tzinfo=timezone.utc)
        except Exception:
            logger.error(f"Aviation Edge: bad date '{date_mmddyyyy}'")
            return None

        days_delta = (flight_date.date() - datetime.utcnow().date()).days
        iso_date = flight_date.strftime("%Y-%m-%d")

        # Extract airline + flight number
        airline, number, _sfx = _split_ident(ident)

        # Strategy 1: Real-time timetable (for recent/today flights)
        if -2 <= days_delta <= 2 and origin_hint:
            params = {
                "iataCode": origin_hint,
                "type": "departure",
            }
            status, data = await self._get("timetable", params)

            if status == 200 and isinstance(data, list):
                # Filter by flight number
                for item in data:
                    flight_iata = (
                        (item.get("flight") or {}).get("iataNumber", "").upper()
                    )
                    # Match DL9013 or just 9013
                    if flight_iata == ident or (
                        number and flight_iata.endswith(number)
                    ):
                        return self._normalize_timetable(item)

        # Strategy 2: Future schedules (for dates >2 days out)
        if days_delta > 2:
            params = {
                "type": "departure",
                "iataCode": origin_hint
                or dest_hint
                or "JFK",  # Need at least one airport
                "date": iso_date,
            }
            status, data = await self._get("flightsFuture", params)

            if status == 200 and isinstance(data, list):
                for item in data:
                    flight_iata = (
                        (item.get("flight") or {}).get("iataNumber", "").upper()
                    )
                    if flight_iata == ident or (
                        number and flight_iata.endswith(number)
                    ):
                        return self._normalize_future_schedule(item)

        # Strategy 3: Routes fallback (if we have origin + dest)
        if origin_hint and dest_hint:
            params = {
                "departureIata": origin_hint,
                "arrivalIata": dest_hint,
            }
            if airline:
                params["airlineIata"] = airline

            status, data = await self._get("routes", params)

            if status == 200 and isinstance(data, list):
                # Routes don't have dates, but validate the route exists
                for item in data:
                    flight_num = str(item.get("flightNumber", "")).upper()
                    if number and flight_num == number:
                        return self._normalize_route(item, iso_date)

        return None

    def _normalize_timetable(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Convert /timetable response to standard format."""
        return {
            "origin": _normalize_airport_fast(
                (item.get("departure") or {}).get("iataCode")
            ),
            "destination": _normalize_airport_fast(
                (item.get("arrival") or {}).get("iataCode")
            ),
            "scheduled_out": (item.get("departure") or {}).get("scheduledTime"),
            "scheduled_in": (item.get("arrival") or {}).get("scheduledTime"),
            "flight": item.get("flight"),
            "airline": item.get("airline"),
            "status": item.get("status"),
        }

    def _normalize_future_schedule(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Convert /flightsFuture response to standard format."""
        dep = item.get("departure") or {}
        arr = item.get("arrival") or {}

        return {
            "origin": _normalize_airport_fast(dep.get("iataCode")),
            "destination": _normalize_airport_fast(arr.get("iataCode")),
            "scheduled_out": None,  # Future schedules don't have exact times
            "scheduled_in": None,
            "flight": item.get("flight"),
            "airline": item.get("airline"),
        }

    def _normalize_route(self, item: Dict[str, Any], date_iso: str) -> Dict[str, Any]:
        """Convert /routes response to standard format."""
        return {
            "origin": _normalize_airport_fast(item.get("departureIata")),
            "destination": _normalize_airport_fast(item.get("arrivalIata")),
            "scheduled_out": None,
            "scheduled_in": None,
            "flight": {
                "iataNumber": f"{item.get('airlineIata', '')}{item.get('flightNumber', '')}"
            },
            "airline": {"iataCode": item.get("airlineIata")},
        }


# ─────────────────────────────────────────────────────────────────────────────
# VALIDATOR
# ─────────────────────────────────────────────────────────────────────────────


class FastFlightValidator:
    """
    Validates & enriches each extracted flight.
    Now with cargo-specific Aviation Edge support!
    """

    def __init__(self) -> None:
        self._has_aero = bool(FLIGHTAWARE_API_KEY)
        self._has_fr24 = bool(FLIGHTRADAR24_API_KEY)

        self._aero: Optional[AeroAPIClient] = (
            AeroAPIClient(FLIGHTAWARE_API_KEY) if self._has_aero else None
        )
        self._fr24: Optional[FR24Client] = (
            FR24Client(FLIGHTRADAR24_API_KEY) if self._has_fr24 else None
        )

        logger.info("🔧 Validator initialized:")
        logger.info(f"   FlightAware API: {'✅' if self._has_aero else '❌'}")
        logger.info(f"   FR24 API: {'✅' if self._has_fr24 else '❌'}")

    async def __aenter__(self) -> "FastFlightValidator":
        tasks = []
        if self._aero:
            tasks.append(self._aero.__aenter__())
        if self._fr24:
            tasks.append(self._fr24.__aenter__())
        if tasks:
            await asyncio.gather(*tasks)
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        tasks = []
        if self._aero:
            tasks.append(self._aero.__aexit__(None, None, None))
        if self._fr24:
            tasks.append(self._fr24.__aexit__(None, None, None))
        if tasks:
            await asyncio.gather(*tasks)

    def _is_cargo_flight(self, flight_no: str) -> bool:
        airline, _num, _sfx = _split_ident(flight_no.upper())

        # 🔥 ADD THIS DEBUG LOGGING:
        is_cargo = airline in CARGO_CARRIERS if airline else False
        logger.info(
            f"   🔍 Cargo detection: '{flight_no}' -> airline='{airline}' -> is_cargo={is_cargo}"
        )
        if airline and not is_cargo:
            logger.info(
                f"      Available cargo carriers: {sorted(CARGO_CARRIERS)[:10]}"
            )

        return is_cargo

    def _cache_get(self, flight_no: str, date: str, flight: Dict[str, Any]) -> Optional[ValidationResult]:
        """Cache key now includes data completeness to avoid wrong hits."""
        # Include completeness in cache key
        has_origin = bool(flight.get("origin"))
        has_dest = bool(flight.get("dest"))
        has_time = bool(flight.get("sched_out_local"))
        
        key = (flight_no.upper(), date, has_origin, has_dest, has_time)
        entry = _VALIDATION_CACHE.get(key)
        if entry and (time.time() - entry[0] < _CACHE_TTL_S):
            logger.info(f"   📦 Cache hit for {flight_no} on {date}")
            return entry[1]
        return None

    def _cache_put(self, flight_no: str, date: str, flight: Dict[str, Any], result: "ValidationResult") -> None:
        """Cache key now includes data completeness."""
        has_origin = bool(flight.get("origin"))
        has_dest = bool(flight.get("dest"))
        has_time = bool(flight.get("sched_out_local"))
        
        key = (flight_no.upper(), date, has_origin, has_dest, has_time)
        _VALIDATION_CACHE[key] = (time.time(), result)
        if len(_VALIDATION_CACHE) > _CACHE_MAX:
            for _ in range(max(1, _CACHE_MAX // 10)):
                try:
                    _VALIDATION_CACHE.pop(next(iter(_VALIDATION_CACHE)))
                except Exception:
                    break

    async def _validate_one(self, flight: Dict[str, Any]) -> List[ValidationResult]:
        """
        Now returns LIST of ValidationResults.
        - Single item list for normal flights
        - Multiple items when multiple matches found
        """
        flight_no = (flight.get("flight_no") or "").strip().upper()
        date = (flight.get("date") or "").strip()

        logger.info(f"🔍 Validating {flight_no} on {date}")

        if not flight_no or not date:
            return [ValidationResult(
                is_valid=False,
                confidence=0.0,
                source="none",
                warnings=["missing_fields"],
            )]

        # Cache check
        cached = self._cache_get(flight_no, date, flight)
        if cached:
            return [cached]  # 🔥 Return as list

        # Fast pass if already complete
        if all(
            flight.get(k)
            for k in ("origin", "dest", "sched_out_local", "sched_in_local")
        ):
            res = ValidationResult(is_valid=True, confidence=1.0, source="complete")
            self._cache_put(flight_no, date, flight, res)
            return [res]  # 🔥 Return as list

        filled: Dict[str, Any] = {}
        source_tags: List[str] = []
        warnings: List[str] = []
        conf = 0.0

        # 🎯 ROUTING LOGIC
        is_cargo = self._is_cargo_flight(flight_no)

        if is_cargo:
            logger.info(f"   🚛 Cargo flight - priority: FR24 → FlightAware")
            priority_order = ["fr24", "aeroapi"]
        else:
            logger.info(f"   ✈️  Passenger flight - priority: FlightAware → FR24")
            priority_order = ["aeroapi", "fr24"]

        # API calls
        api_results: List[Tuple[str, Optional[Union[Dict[str, Any], List[Dict[str, Any]]]]]] = []

        async def _call_aero() -> None:
            if not self._aero:
                api_results.append(("aeroapi", None))
                return
            data = await self._aero.search(
                flight_no,
                date,
                origin_hint=_normalize_airport_fast(flight.get("origin")),
                dest_hint=_normalize_airport_fast(flight.get("dest")),
            )
            api_results.append(("aeroapi", data))

        async def _call_fr24() -> None:
            if not self._fr24:
                api_results.append(("fr24", None))
                return
            data = await self._fr24.search(flight_no, date)
            api_results.append(("fr24", data))

        # Call both APIs in parallel
        await asyncio.gather(_call_aero(), _call_fr24())

        # Sort by priority
        api_results_sorted = sorted(
            api_results,
            key=lambda x: priority_order.index(x[0]) if x[0] in priority_order else 999,
        )

        # Process results
        for tag, data in api_results_sorted:
            if not data:
                continue

            # 🔥 HANDLE MULTIPLE RESULTS - RETURN ALL!
            if isinstance(data, list) and len(data) > 1:
                logger.info(f"   📋 {tag} returned {len(data)} options - RETURNING ALL")
                
                # Process ALL options
                all_results = []
                for idx, item in enumerate(data):
                    filled_temp: Dict[str, Any] = {}
                    warnings_temp: List[str] = []
                    
                    if tag == "aeroapi":
                        self._apply_aero_fields(item, flight, filled_temp, warnings_temp)
                    elif tag == "fr24":
                        self._apply_fr24_fields(item, flight, filled_temp, warnings_temp)
                    
                    if filled_temp:
                        # Create separate ValidationResult for each option
                        conf_temp = 0.95 if tag == "aeroapi" else 0.85
                        warnings_temp.append(f"Option {idx + 1} of {len(data)}")
                        
                        all_results.append(ValidationResult(
                            is_valid=True,
                            confidence=conf_temp * 0.8,  # Lower confidence for multi-match
                            source=f"{tag}",
                            filled_fields=filled_temp,
                            warnings=warnings_temp,
                        ))
                
                if all_results:
                    logger.info(f"   ✅ Returning {len(all_results)} options to frontend")
                    return all_results  # 🔥 RETURN ALL OPTIONS
                
                # If no valid options, continue to next API
                continue

            # Handle single result
            if isinstance(data, list):
                data = data[0] if data else None
                if not data:
                    continue

            # Track fields before
            fields_before = set(filled.keys())

            if tag == "aeroapi":
                self._apply_aero_fields(data, flight, filled, warnings)
                if len(filled) > len(fields_before):
                    source_tags.append("aeroapi")
                    conf = max(conf, 0.95)
            elif tag == "fr24":
                self._apply_fr24_fields(data, flight, filled, warnings)
                if len(filled) > len(fields_before):
                    source_tags.append("fr24")
                    conf = max(conf, 0.85)

            # Early exit if complete
            has_all = all(
                filled.get(k) or flight.get(k)
                for k in ("origin", "dest", "sched_out_local", "sched_in_local")
            )
            if has_all:
                logger.info(f"   ✅ All fields complete after {tag}")
                break

        if not source_tags:
            res = ValidationResult(
                is_valid=True,
                confidence=0.5,
                source="heuristic",
                filled_fields={},
                warnings=["No API data found"]
            )
            self._cache_put(flight_no, date, flight, res)
            return [res]  # 🔥 Return as list

        res = ValidationResult(
            is_valid=True,
            confidence=conf if conf > 0 else 0.5,
            source="+".join(source_tags),
            filled_fields=filled,
            warnings=warnings,
        )
        self._cache_put(flight_no, date, flight, res)
        return [res]  # 🔥 Return as list

    def _apply_aviedge_fields(self, api_data: Dict[str, Any], flight: Dict[str, Any], filled: Dict[str, Any]) -> None:
        """
        Extract fields from Aviation Edge response.
        Handles /timetable, /flightsFuture, and /routes responses.
        """
        # Origin/Dest
        origin = _normalize_airport_fast(api_data.get("origin"))
        dest = _normalize_airport_fast(api_data.get("destination"))
        
        if origin and not flight.get("origin"):
            filled["origin"] = origin
        if dest and not flight.get("dest"):
            filled["dest"] = dest

        # Times (if present - /timetable has them, /flightsFuture doesn't)
        sched_out = api_data.get("scheduled_out")
        sched_in = api_data.get("scheduled_in")
        
        # 🔥 CONVERT API TIMES FIRST
        api_out_local = None
        api_in_local = None
        
        if sched_out:
            api_out_local = _to_local_hhmm_from_iso(sched_out, origin or flight.get("origin"))
        if sched_in:
            api_in_local = _to_local_hhmm_from_iso(sched_in, dest or flight.get("dest"))
        
        # 🔥 VALIDATE: If user provided departure time, API must match within 30 minutes
        user_time = flight.get("sched_out_local")
        if user_time and api_out_local:
            try:
                user_minutes = int(user_time[:2]) * 60 + int(user_time[2:])
                api_minutes = int(api_out_local[:2]) * 60 + int(api_out_local[2:])
                time_diff = abs(api_minutes - user_minutes)
                
                if time_diff > 30:  # More than 30 minutes difference
                    logger.warning(f"   ⚠️  Time mismatch: user={user_time} api={api_out_local} (diff={time_diff}min) - REJECTING")
                    # Don't use this flight data at all
                    return
                else:
                    logger.info(f"   ✅ Time validated: user={user_time} api={api_out_local} (diff={time_diff}min)")
            except Exception as e:
                logger.warning(f"   ⚠️  Time validation error: {e}")
        
        # 🔥 ONLY FILL IF VALIDATION PASSED
        if api_out_local and not flight.get("sched_out_local"):
            filled["sched_out_local"] = api_out_local
        
        if api_in_local and not flight.get("sched_in_local"):
            filled["sched_in_local"] = api_in_local

    def _apply_aero_fields(
        self, api_data: Dict[str, Any], flight: Dict[str, Any], filled: Dict[str, Any], warnings: List[str]
    ) -> None:
        """
        Extract origin/dest + scheduled times from AeroAPI record.
        Works with both /flights and /schedules shapes.
        """
        # Origin / Dest - add the _iata variants used by /schedules
        origin = (
            _normalize_airport_fast(api_data.get("origin"))
            or _normalize_airport_fast(api_data.get("origin_iata"))
            or _normalize_airport_fast(api_data.get("departure_airport"))
        )
        dest = (
            _normalize_airport_fast(api_data.get("destination"))
            or _normalize_airport_fast(api_data.get("destination_iata"))
            or _normalize_airport_fast(api_data.get("arrival_airport"))
        )
        if not origin:
            # some schedules nest airports
            origin = _normalize_airport_fast(
                (api_data.get("departure") or {}).get("airport")
            )
        if not dest:
            dest = _normalize_airport_fast(
                (api_data.get("arrival") or {}).get("airport")
            )
        if origin and not flight.get("origin"):
            filled["origin"] = origin
        if dest and not flight.get("dest"):
            filled["dest"] = dest

        # Times - helper to extract ISO timestamps
        def _first_iso(*keys: str) -> Optional[str]:
            for k in keys:
                v = api_data.get(k)
                if isinstance(v, str) and ("T" in v or v.endswith("Z")):
                    return v
                if isinstance(v, dict):
                    sched = v.get("scheduled")
                    if isinstance(sched, str) and ("T" in sched or sched.endswith("Z")):
                        return sched
            return None

        # 🔥 STEP 1: Extract and convert API times
        iso_dep = _first_iso(
            "scheduled_out", "scheduled_off", "departure_time", "departure"
        )
        iso_arr = _first_iso(
            "scheduled_in", "scheduled_on", "arrival_time", "arrival"
        )
        
        api_out_local = None
        api_in_local = None
        
        
        if iso_dep:
            api_out_local = _to_local_hhmm_from_iso(iso_dep, filled.get("origin") or flight.get("origin"))
        if iso_arr:
            api_in_local = _to_local_hhmm_from_iso(iso_arr, filled.get("dest") or flight.get("dest"))

        # 🔥 SMARTER time validation
        user_time = flight.get("sched_out_local")
        if user_time and api_out_local:
            try:
                user_minutes = int(user_time[:2]) * 60 + int(user_time[2:])
                api_minutes = int(api_out_local[:2]) * 60 + int(api_out_local[2:])
                time_diff = abs(api_minutes - user_minutes)

                if time_diff > 120:  # More than 2 hours - definitely wrong flight
                    logger.warning(
                        f"   ⚠️  Time mismatch: user={user_time} api={api_out_local} "
                        f"(diff={time_diff}min) - REJECTING (>2hrs)"
                    )
                    filled.clear()
                    return
                elif time_diff > 15:  # 15min-2hrs - possibly correct but uncertain
                    logger.warning(
                        f"   ⚠️  Time mismatch: user={user_time} api={api_out_local} "
                        f"(diff={time_diff}min) - UNCERTAIN"
                    )
                    warnings.append(f"Departure time differs by {time_diff} minutes from expected")
                else:
                    logger.info(
                        f"   ✅ Time validated: user={user_time} api={api_out_local} (diff={time_diff}min)"
                    )
            except Exception as e:
                logger.warning(f"   ⚠️  Time validation error: {e}")

        # 🔥 STEP 3: Fill times only if validation passed (or no user hint)
        if api_out_local and not flight.get("sched_out_local"):
            filled["sched_out_local"] = api_out_local

        if api_in_local and not flight.get("sched_in_local"):
            filled["sched_in_local"] = api_in_local

    def _apply_fr24_fields(self, api_data: Dict[str, Any], flight: Dict[str, Any], filled: Dict[str, Any], warnings: List[str]) -> None:
        """
        Extract fields from FR24 record.
        """
        airport = api_data.get("airport", {}) if isinstance(api_data, dict) else {}
        o_code = airport.get("origin", {}).get("code", {})
        d_code = airport.get("destination", {}).get("code", {})

        origin = _normalize_airport_fast(o_code.get("iata") or o_code.get("icao") or o_code)
        dest = _normalize_airport_fast(d_code.get("iata") or d_code.get("icao") or d_code)

        if origin and not flight.get("origin"):
            filled["origin"] = origin
        if dest and not flight.get("dest"):
            filled["dest"] = dest

        times = api_data.get("time", {}).get("scheduled", {})
        dep_epoch = times.get("departure")
        arr_epoch = times.get("arrival")

        # 🔥 STEP 1: Convert FR24 epoch times to local HHMM
        api_out_local = None
        api_in_local = None
        
        if dep_epoch:
            api_out_local = _to_local_hhmm_from_epoch_utc(
                dep_epoch, filled.get("origin") or flight.get("origin")
            )
        if arr_epoch:
            api_in_local = _to_local_hhmm_from_epoch_utc(
                arr_epoch, filled.get("dest") or flight.get("dest")
            )

        # 🔥 SMARTER time validation (same logic as aero)
        user_time = flight.get("sched_out_local")
        if user_time and api_out_local:
            try:
                user_minutes = int(user_time[:2]) * 60 + int(user_time[2:])
                api_minutes = int(api_out_local[:2]) * 60 + int(api_out_local[2:])
                time_diff = abs(api_minutes - user_minutes)

                if time_diff > 120:
                    logger.warning(
                        f"   ⚠️  FR24 time mismatch: user={user_time} api={api_out_local} "
                        f"(diff={time_diff}min) - REJECTING (>2hrs)"
                    )
                    filled.clear()
                    return
                elif time_diff > 15:
                    logger.warning(
                        f"   ⚠️  FR24 time mismatch: user={user_time} api={api_out_local} "
                        f"(diff={time_diff}min) - UNCERTAIN"
                    )
                    warnings.append(f"Departure time differs by {time_diff} minutes from expected")
                else:
                    logger.info(
                        f"   ✅ FR24 time validated: user={user_time} api={api_out_local} (diff={time_diff}min)"
                    )
            except Exception as e:
                logger.warning(f"   ⚠️  FR24 time validation error: {e}")

        # 🔥 STEP 3: Fill times only if validation passed
        if api_out_local and not flight.get("sched_out_local"):
            filled["sched_out_local"] = api_out_local

        if api_in_local and not flight.get("sched_in_local"):
            filled["sched_in_local"] = api_in_local
        
    async def validate_batch(self, flights: List[Dict[str, Any]]) -> Dict[str, Any]:
        start = time.time()
        need = []
        prefilled = []
        
        for f in flights:
            if all(f.get(k) for k in ("origin", "dest", "sched_out_local", "sched_in_local")):
                prefilled.append([ValidationResult(is_valid=True, confidence=1.0, source="complete")])
            else:
                need.append(f)

        logger.info("🚀 BATCH VALIDATION STARTED")
        logger.info(f"   Total flights: {len(flights)} | Need validation: {len(need)}")

        validations: List[List[ValidationResult]] = []  # 🔥 Now list of lists
        
        if need:
            for i, f in enumerate(need):
                try:
                    if i > 0 and i % 3 == 0:
                        await asyncio.sleep(1.0)
                        
                    results = await self._validate_one(f)  # 🔥 Returns list now
                    validations.append(results)
                except Exception as e:
                    logger.error(f"Validation exception: {e}")
                    validations.append([ValidationResult(
                        is_valid=False,
                        confidence=0.0,
                        source="error",
                        warnings=[str(e)]
                    )])

        # 🔥 Flatten: Create one EnrichedFlight per ValidationResult
        enriched: List[EnrichedFlight] = []
        
        for raw, val_list in zip(flights, validations if validations else prefilled):
            for val in val_list:  # 🔥 Loop through all results for this flight
                ef = EnrichedFlight(**raw, validation_result=val)
                for k, v in (val.filled_fields or {}).items():
                    setattr(ef, k, v)
                enriched.append(ef)

        valid_count = sum(1 for e in enriched if e.validation_result and e.validation_result.is_valid)
        avg_conf = (
            (sum(e.validation_result.confidence for e in enriched if e.validation_result) / len(enriched))
            if enriched else 0.0
        )
        total_filled = sum(len(e.validation_result.filled_fields) for e in enriched if e.validation_result)
        elapsed = time.time() - start

        logger.info("✅ BATCH VALIDATION COMPLETE")
        logger.info(f"   Enriched flights: {len(enriched)} from {len(flights)} input | {elapsed:.2f}s")

        return {
            "enriched_flights": [e.dict() for e in enriched],
            "validation_summary": {
                "total_input_flights": len(flights),
                "total_output_flights": len(enriched),
                "valid_flights": valid_count,
                "average_confidence": avg_conf,
                "total_fields_filled": total_filled,
                "processing_time_seconds": elapsed,
                "sources_used": sorted(set((e.validation_result.source if e.validation_result else "none") for e in enriched)),
            },
        }


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC ENTRYPOINT
# ─────────────────────────────────────────────────────────────────────────────


async def validate_extraction_results(
    extraction_result: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Accepts your extractor output and returns the same dict updated with:
      - "validation": {...}
      - "enriched_flights": [...]
    """
    logger.info("=" * 60)
    logger.info("VALIDATION REQUEST RECEIVED")
    logger.info("=" * 60)

    flights = extraction_result.get("flights", [])
    async with FastFlightValidator() as validator:
        summary = await validator.validate_batch(flights)

    extraction_result["validation"] = summary["validation_summary"]
    extraction_result["enriched_flights"] = summary["enriched_flights"]

    logger.info("=" * 60)
    logger.info("VALIDATION REQUEST COMPLETE")
    logger.info("=" * 60)
    return extraction_result


__all__ = [
    "validate_extraction_results",
    "FastFlightValidator",
    "AeroAPIClient",
    "FR24Client",
    "ValidationResult",
    "EnrichedFlight",
]
