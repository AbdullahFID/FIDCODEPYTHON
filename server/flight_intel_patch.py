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
FR24_BASE_URL = "https://api.flightradar24.com/common/v1"     # FR24 public app API

# Rate limiting / concurrency knobs
AEROAPI_MAX_RPS = float(os.getenv("AEROAPI_MAX_RPS", "10"))   # steady tokens per second
AEROAPI_BURST = int(os.getenv("AEROAPI_BURST", "3"))         # bucket capacity
VALIDATOR_CONCURRENCY = int(os.getenv("VALIDATOR_CONCURRENCY", "1"))

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
_IDENT_SPLIT_RE: Pattern[str] = _re.compile(r"^([A-Z]{2,3})?(\d{1,5})([A-Z]?)$")

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
    "JFK": "America/New_York", "LGA": "America/New_York", "EWR": "America/New_York",
    "ATL": "America/New_York", "BOS": "America/New_York", "DCA": "America/New_York",
    "MIA": "America/New_York", "MCO": "America/New_York", "TPA": "America/New_York",
    "PHL": "America/New_York", "CLT": "America/New_York", "BWI": "America/New_York",
    "DTW": "America/Detroit", "JAX": "America/New_York",
    # Central
    "ORD": "America/Chicago", "MDW": "America/Chicago", "DFW": "America/Chicago",
    "IAH": "America/Chicago", "MSP": "America/Chicago", "STL": "America/Chicago",
    "MCI": "America/Chicago", "MKE": "America/Chicago", "MSY": "America/Chicago",
    "BNA": "America/Chicago", "AUS": "America/Chicago", "SAT": "America/Chicago",
    # Mountain
    "DEN": "America/Denver", "SLC": "America/Denver", "PHX": "America/Phoenix",
    # Pacific
    "LAX": "America/Los_Angeles", "SFO": "America/Los_Angeles", "SEA": "America/Los_Angeles",
    "SAN": "America/Los_Angeles", "PDX": "America/Los_Angeles", "LAS": "America/Los_Angeles",
    "SJC": "America/Los_Angeles", "OAK": "America/Los_Angeles", "SMF": "America/Los_Angeles",
    # Alaska/Hawaii
    "ANC": "America/Anchorage", "HNL": "Pacific/Honolulu",
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

def _to_local_hhmm_from_epoch_utc(epoch: Optional[int], airport_iata: Optional[str]) -> Optional[str]:
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
    ) -> Optional[Dict[str, Any]]:
        """
        Returns a single best-match flight dict or None.
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

        day_start_iso, day_end_iso = _iso_day_window_utc(flight_dt.year, flight_dt.month, flight_dt.day)
        days_delta = (flight_dt.date() - datetime.utcnow().date()).days

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
                logger.info(f"AeroAPI GET (cache) {path} params={params} status={status}")
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
                    logger.info(f"AeroAPI GET {path} params={params} status={status} took={elapsed:.2f}s")

                    if status == 429 and attempt < attempts - 1:
                        ra = r.headers.get("Retry-After")
                        if ra:
                            try:
                                await asyncio.sleep(float(ra) + 0.5)  # Add extra 0.5s buffer
                            except Exception:
                                await asyncio.sleep(2.0)  # Increase from 0.8
                        else:
                            await asyncio.sleep(2.0 + random.random())  # Increase base delay
                        continue

                    if status == 200:
                        _AERO_RESP_CACHE[key] = (time.perf_counter(), body)
                        if path.startswith("/schedules") and isinstance(body, dict):
                            if "scheduled" in body:
                                logger.info(f"   Found {len(body['scheduled'])} scheduled flights")
                                if body['scheduled']:
                                    first = body['scheduled'][0]
                                    logger.info(f"   First flight: {first.get('ident_iata')} from {first.get('origin_iata')} to {first.get('destination_iata')}")
                        return status, body, elapsed
                    return status, body, elapsed
            return status, body, elapsed

        # Strategy Phase A — live/near/old vs future
        phaseA: List[Tuple[str, Dict[str, Any]]] = []

        if days_delta < -10:
            # Older than ~10 days → /history
            phaseA.append((f"/history/flights/{ident}", {"start": day_start_iso, "end": day_end_iso}))
        elif -10 <= days_delta <= 2:
            # within 2 days → /flights
            phaseA.append((f"/flights/{ident}", {"start": day_start_iso, "end": day_end_iso}))
        else:
            # Future/schedules window → airline+flight_number first
            airline, number, _suffix = _split_ident(ident)
            if number:
                params = {"flight_number": number, **S_PARAMS}
                if airline:
                    params["airline"] = airline
                phaseA.append((
                    f"/schedules/{flight_dt.strftime('%Y-%m-%d')}/{(flight_dt + timedelta(days=1)).strftime('%Y-%m-%d')}",
                    params
                ))

        # Always allow cross-try of flights/history to be safe (won't match far future)
        def _have(prefix: str, arr: List[Tuple[str, Dict[str, Any]]]) -> bool:
            return any(p.startswith(prefix) for p, _ in arr)

        if not _have("/flights/", phaseA):
            phaseA.append((f"/flights/{ident}", {"start": day_start_iso, "end": day_end_iso}))
        if not _have("/history/", phaseA):
            phaseA.append((f"/history/flights/{ident}", {"start": day_start_iso, "end": day_end_iso}))

        last_error_text: Optional[str] = None

        # Helper to parse and pick an item
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
                last_error_text = body if isinstance(body, str) else json.dumps(body)[:200]
                continue

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
            route_tries.append({"airline": airline or "DL", "origin": origin_hint, **S_PARAMS})

        for params in route_tries:
            status, body, _elapsed = await _do_get(sched_path, params)

            if status == 400:
                err_txt = body if isinstance(body, str) else json.dumps(body)[:200]
                logger.error(f"AeroAPI 400 on {sched_path} params={params} err={err_txt}")
                last_error_text = err_txt
                continue
            if status in (404, 204):
                continue
            if status != 200:
                last_error_text = body if isinstance(body, str) else json.dumps(body)[:200]
                continue

            item = _extract_first_item(body)
            if item:
                return item

        if last_error_text:
            logger.error(f"AeroAPI no data; last error: {last_error_text}")
        return None

# ─────────────────────────────────────────────────────────────────────────────
# FLIGHTRADAR24 CLIENT (best-effort)
# ─────────────────────────────────────────────────────────────────────────────

class FR24Client:
    """
    Lightweight FR24 fetcher. Public app endpoints shift occasionally; this is a best-effort helper.
    We only rely on it as a supplemental source.

    Endpoint used:
      - GET /flight/list.json?query={IDENT}&fetchBy=flight&limit=10&token={API_KEY}

    Output (typical):
      { "result": { "response": { "data": [ { "airport": { origin: { code: { iata: "SFO" }}, destination: {...}}, "time": { "scheduled": { "departure": 163..., "arrival": 163... } } } ] } } }
    """

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key
        self._session: Optional[aiohttp.ClientSession] = None
        self._connector = aiohttp.TCPConnector(limit=20, limit_per_host=10, ttl_dns_cache=300)

    async def __aenter__(self) -> "FR24Client":
        timeout = aiohttp.ClientTimeout(total=10, connect=3)
        self._session = aiohttp.ClientSession(
            headers={"Accept": "application/json", "User-Agent": "Flight-Intel/2.3"},
            connector=self._connector,
            timeout=timeout,
        )
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._session:
            await self._session.close()

    async def search(self, flight_no: str, date_mmddyyyy: str) -> Optional[Dict[str, Any]]:
        if not self._session or not self._api_key:
            return None

        ident = flight_no.strip().upper()
        params = {
            "query": ident,
            "fetchBy": "flight",
            "limit": 10,
            "token": self._api_key,
        }
        url = f"{FR24_BASE_URL}/flight/list.json"
        t0 = time.perf_counter()
        try:
            async with self._session.get(url, params=params) as r:
                elapsed = time.perf_counter() - t0
                logger.info(f"FR24 GET list status={r.status} took={elapsed:.2f}s")
                if r.status != 200:
                    txt = await r.text()
                    logger.error(f"FR24 error {r.status}: {txt[:200]}")
                    return None
                data = await r.json()
        except asyncio.TimeoutError:
            logger.error("FR24 timeout")
            return None
        except Exception as e:
            logger.error(f"FR24 exception: {e}")
            return None

        try:
            flights = (
                data.get("result", {})
                    .get("response", {})
                    .get("data") or []
            )
            if flights:
                return flights[0]
        except Exception:
            pass
        return None

# ─────────────────────────────────────────────────────────────────────────────
# AVIATION EDGE CLIENT (Cargo-focused)
# ─────────────────────────────────────────────────────────────────────────────

AVIATION_EDGE_API_KEY: Optional[str] = os.getenv("AVIATION_EDGE_KEY")
AVIATION_EDGE_BASE = "https://aviation-edge.com/v2/public"

# Cargo carriers that should prefer Aviation Edge
CARGO_CARRIERS = {
    "5X", "UPS",  # UPS
    "FX", "FDX",  # FedEx
    "5Y", "GTI",  # Atlas Air
    "K4", "CKS",  # Kalitta Air
    "NC", "NAC",  # Northern Air Cargo
    "GB", "ABX",  # ABX Air
    "3S", "PAC",  # Polar Air Cargo
    "M6", "AJT",  # Amerijet
    "CV", "CLX",  # Cargolux
    "KZ", "NCA",  # Nippon Cargo
    "PO",         # Polar Air Cargo (alternate code)
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
                    logger.info(f"Aviation Edge GET /{endpoint} status={status} results={len(data) if isinstance(data, list) else 'N/A'}")
                    return status, data
                else:
                    text = await r.text()
                    logger.error(f"Aviation Edge GET /{endpoint} status={status} error={text[:200]}")
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
                    flight_iata = (item.get("flight") or {}).get("iataNumber", "").upper()
                    # Match DL9013 or just 9013
                    if flight_iata == ident or (number and flight_iata.endswith(number)):
                        return self._normalize_timetable(item)

        # Strategy 2: Future schedules (for dates >2 days out)
        if days_delta > 2:
            params = {
                "type": "departure",
                "iataCode": origin_hint or dest_hint or "JFK",  # Need at least one airport
                "date": iso_date,
            }
            status, data = await self._get("flightsFuture", params)
            
            if status == 200 and isinstance(data, list):
                for item in data:
                    flight_iata = (item.get("flight") or {}).get("iataNumber", "").upper()
                    if flight_iata == ident or (number and flight_iata.endswith(number)):
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
            "origin": _normalize_airport_fast((item.get("departure") or {}).get("iataCode")),
            "destination": _normalize_airport_fast((item.get("arrival") or {}).get("iataCode")),
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
            "flight": {"iataNumber": f"{item.get('airlineIata', '')}{item.get('flightNumber', '')}"},
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
        self._has_aviedge = bool(AVIATION_EDGE_API_KEY)
        
        self._aero: Optional[AeroAPIClient] = AeroAPIClient(FLIGHTAWARE_API_KEY) if self._has_aero else None
        self._fr24: Optional[FR24Client] = FR24Client(FLIGHTRADAR24_API_KEY) if self._has_fr24 else None
        self._aviedge: Optional[AviationEdgeClient] = AviationEdgeClient(AVIATION_EDGE_API_KEY) if self._has_aviedge else None

        logger.info("🔧 Validator initialized:")
        logger.info(f"   FlightAware API: {'✅' if self._has_aero else '❌'}")
        logger.info(f"   FR24 API: {'✅' if self._has_fr24 else '❌'}")
        logger.info(f"   Aviation Edge API: {'✅' if self._has_aviedge else '❌'}")

    async def __aenter__(self) -> "FastFlightValidator":
        tasks = []
        if self._aero:
            tasks.append(self._aero.__aenter__())
        if self._fr24:
            tasks.append(self._fr24.__aenter__())
        if self._aviedge:
            tasks.append(self._aviedge.__aenter__())
        if tasks:
            await asyncio.gather(*tasks)
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        tasks = []
        if self._aero:
            tasks.append(self._aero.__aexit__(None, None, None))
        if self._fr24:
            tasks.append(self._fr24.__aexit__(None, None, None))
        if self._aviedge:
            tasks.append(self._aviedge.__aexit__(None, None, None))
        if tasks:
            await asyncio.gather(*tasks)

    def _is_cargo_flight(self, flight_no: str) -> bool:
        """Detect if flight is cargo based on airline code."""
        airline, _num, _sfx = _split_ident(flight_no.upper())
        return airline in CARGO_CARRIERS if airline else False

    def _cache_get(self, flight_no: str, date: str) -> Optional[ValidationResult]:
        key = (flight_no.upper(), date)
        entry = _VALIDATION_CACHE.get(key)
        if entry and (time.time() - entry[0] < _CACHE_TTL_S):
            logger.info(f"   📦 Cache hit for {key}")
            return entry[1]
        return None

    def _cache_put(self, flight_no: str, date: str, result: "ValidationResult") -> None:
        key = (flight_no.upper(), date)
        _VALIDATION_CACHE[key] = (time.time(), result)
        if len(_VALIDATION_CACHE) > _CACHE_MAX:
            # drop ~10% oldest
            for _ in range(max(1, _CACHE_MAX // 10)):
                try:
                    _VALIDATION_CACHE.pop(next(iter(_VALIDATION_CACHE)))
                except Exception:
                    break

    async def _validate_one(self, flight: Dict[str, Any]) -> ValidationResult:
        flight_no = (flight.get("flight_no") or "").strip().upper()
        date = (flight.get("date") or "").strip()

        logger.info(f"🔍 Validating {flight_no} on {date}")

        if not flight_no or not date:
            return ValidationResult(is_valid=False, confidence=0.0, source="none", warnings=["missing_fields"])

        # Cache check
        cached = self._cache_get(flight_no, date)
        if cached:
            return cached

        # Fast pass if already complete
        if all(flight.get(k) for k in ("origin", "dest", "sched_out_local", "sched_in_local")):
            res = ValidationResult(is_valid=True, confidence=1.0, source="complete")
            self._cache_put(flight_no, date, res)
            return res

        filled: Dict[str, Any] = {}
        source_tags: List[str] = []
        conf = 0.0

        # 🎯 ROUTING LOGIC: Determine priority order
        is_cargo = self._is_cargo_flight(flight_no)
        
        if is_cargo:
            logger.info(f"   🚛 Cargo flight detected - priority: Aviation Edge → FlightAware → FR24")
        else:
            logger.info(f"   ✈️  Passenger flight detected - priority: FlightAware → Aviation Edge → FR24")

        # Prepare API calls with priority order
        api_results: List[Tuple[str, Optional[Dict[str, Any]]]] = []

        async def _call_aviedge() -> None:
            """Call Aviation Edge API."""
            if not self._aviedge:
                api_results.append(("aviedge", None))
                return
            data = await self._aviedge.search(
                flight_no,
                date,
                origin_hint=_normalize_airport_fast(flight.get("origin")),
                dest_hint=_normalize_airport_fast(flight.get("dest")),
            )
            api_results.append(("aviedge", data))

        async def _call_aero() -> None:
            """Call FlightAware API."""
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
            """Call FR24 API."""
            if not self._fr24:
                api_results.append(("fr24", None))
                return
            data = await self._fr24.search(flight_no, date)
            api_results.append(("fr24", data))

        # 🔥 NEW: Call ALL available APIs in parallel
        # We'll process them in priority order afterwards
        await asyncio.gather(_call_aviedge(), _call_aero(), _call_fr24())

        # 🎯 Process results in priority order based on flight type
        if is_cargo:
            # Cargo priority: Aviation Edge > FlightAware > FR24
            priority_order = ["aviedge", "aeroapi", "fr24"]
        else:
            # Passenger priority: FlightAware > Aviation Edge > FR24
            priority_order = ["aeroapi", "aviedge", "fr24"]

        # Sort api_results by priority
        api_results_sorted = sorted(
            api_results,
            key=lambda x: priority_order.index(x[0]) if x[0] in priority_order else 999
        )

        # Process in priority order, stop early if we get complete data
        for tag, data in api_results_sorted:
            if not data:
                continue

            # Track what we had before this API call
            fields_before = set(filled.keys())

            if tag == "aviedge":
                self._apply_aviedge_fields(data, flight, filled)
                if len(filled) > len(fields_before):  # New fields added
                    source_tags.append("aviation_edge")
                    conf = max(conf, 0.90)
            elif tag == "aeroapi":
                self._apply_aero_fields(data, flight, filled)
                if len(filled) > len(fields_before):  # New fields added
                    source_tags.append("aeroapi")
                    conf = max(conf, 0.95)
            elif tag == "fr24":
                self._apply_fr24_fields(data, flight, filled)
                if len(filled) > len(fields_before):  # New fields added
                    source_tags.append("fr24")
                    conf = max(conf, 0.85)

            # 🚀 EARLY EXIT: If we have all fields, stop calling more APIs
            has_all = all(
                filled.get(k) or flight.get(k) 
                for k in ("origin", "dest", "sched_out_local", "sched_in_local")
            )
            if has_all:
                logger.info(f"   ✅ All fields complete after {tag} - skipping remaining APIs")
                break

        if not source_tags:
            # Heuristic "still valid but low confidence"
            res = ValidationResult(is_valid=True, confidence=0.5, source="heuristic", filled_fields={})
            self._cache_put(flight_no, date, res)
            return res

        res = ValidationResult(
            is_valid=True,
            confidence=conf if conf > 0 else 0.5,
            source="+".join(source_tags),
            filled_fields=filled,
        )
        self._cache_put(flight_no, date, res)
        return res

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
        
        if sched_out and not flight.get("sched_out_local"):
            # Aviation Edge times are ISO strings
            filled["sched_out_local"] = _to_local_hhmm_from_iso(sched_out, origin or flight.get("origin"))
        
        if sched_in and not flight.get("sched_in_local"):
            filled["sched_in_local"] = _to_local_hhmm_from_iso(sched_in, dest or flight.get("dest"))

    def _apply_aero_fields(self, api_data: Dict[str, Any], flight: Dict[str, Any], filled: Dict[str, Any]) -> None:
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

        # Times
        # Possible keys (strings, ISO 8601): scheduled_out, scheduled_in, scheduled_off, scheduled_on
        # Some schedules carry times under departure/arrival dicts (e.g., {"scheduled": "2025-09-19T12:34:00Z"})
        def _first_iso(*keys: str) -> Optional[str]:
            for k in keys:
                v = api_data.get(k)
                if isinstance(v, str) and ("T" in v or v.endswith("Z")):  # Better ISO check
                    return v
                if isinstance(v, dict):
                    sched = v.get("scheduled")
                    if isinstance(sched, str) and ("T" in sched or sched.endswith("Z")):
                        return sched
            return None

        if not flight.get("sched_out_local"):
            iso_dep = _first_iso("scheduled_out", "scheduled_off", "departure_time", "departure")
            if iso_dep:
                filled["sched_out_local"] = _to_local_hhmm_from_iso(iso_dep, filled.get("origin") or flight.get("origin"))

        if not flight.get("sched_in_local"):
            iso_arr = _first_iso("scheduled_in", "scheduled_on", "arrival_time", "arrival")
            if iso_arr:
                filled["sched_in_local"] = _to_local_hhmm_from_iso(iso_arr, filled.get("dest") or flight.get("dest"))

    def _apply_fr24_fields(self, api_data: Dict[str, Any], flight: Dict[str, Any], filled: Dict[str, Any]) -> None:
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

        if not flight.get("sched_out_local"):
            # Convert from epoch(UTC) → local HHMM using origin tz
            hhmm = _to_local_hhmm_from_epoch_utc(dep_epoch, filled.get("origin") or flight.get("origin"))
            if hhmm:
                filled["sched_out_local"] = hhmm

        if not flight.get("sched_in_local"):
            hhmm = _to_local_hhmm_from_epoch_utc(arr_epoch, filled.get("dest") or flight.get("dest"))
            if hhmm:
                filled["sched_in_local"] = hhmm

    async def validate_batch(self, flights: List[Dict[str, Any]]) -> Dict[str, Any]:
        start = time.time()
        need = []
        prefilled = []
        for f in flights:
            if all(f.get(k) for k in ("origin", "dest", "sched_out_local", "sched_in_local")):
                prefilled.append(ValidationResult(is_valid=True, confidence=1.0, source="complete"))
            else:
                need.append(f)

        logger.info("🚀 BATCH VALIDATION STARTED")
        logger.info(f"   Total flights: {len(flights)} | Need validation: {len(need)} | Already complete: {len(prefilled)}")

        validations: List[ValidationResult] = []
        if need:
            # Sequential processing with delay between requests
            for i, f in enumerate(need):
                try:
                    # Add delay after every 3rd request to avoid rate limiting
                    if i > 0 and i % 3 == 0:
                        await asyncio.sleep(1.0)  # 1 second delay every 3 requests
                        
                    result = await self._validate_one(f)
                    validations.append(result)
                except Exception as e:
                    logger.error(f"Validation exception: {e}")
                    validations.append(ValidationResult(
                        is_valid=False, 
                        confidence=0.0, 
                        source="error", 
                        warnings=[str(e)]
                    ))

        # Interleave back into original order
        all_results: List[ValidationResult] = []
        it = iter(validations)
        for f in flights:
            if all(f.get(k) for k in ("origin", "dest", "sched_out_local", "sched_in_local")):
                all_results.append(prefilled.pop(0))
            else:
                all_results.append(next(it))

        enriched: List[EnrichedFlight] = []
        for raw, val in zip(flights, all_results):
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
        logger.info(f"   Valid flights: {valid_count}/{len(flights)} | Avg conf: {avg_conf:.2f} | Fields filled: {total_filled} | {elapsed:.2f}s")

        return {
            "enriched_flights": [e.dict() for e in enriched],
            "validation_summary": {
                "total_flights": len(flights),
                "valid_flights": valid_count,
                "average_confidence": avg_conf,
                "total_fields_filled": total_filled,
                "processing_time_seconds": elapsed,
                "sources_used": sorted(set((e.validation_result.source if e.validation_result else "none") for e in enriched)),
                "warnings": [],
            },
        }
# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC ENTRYPOINT
# ─────────────────────────────────────────────────────────────────────────────

async def validate_extraction_results(extraction_result: Dict[str, Any]) -> Dict[str, Any]:
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


__all__ = ["validate_extraction_results", "FastFlightValidator", "AeroAPIClient", "FR24Client", "ValidationResult", "EnrichedFlight"]
