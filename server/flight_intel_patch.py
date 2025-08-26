# flight_intel_patch.py
# Flight Intel Validation & Enrichment Module – SPEED OPTIMIZED v2.2
#
# What’s new vs your v2.0:
# - FlightAware schedules endpoint uses correct filters (ident OR airline+flight_number), not `flight_ident`
# - Properly passes query params for every endpoint
# - Stronger parsing for AeroAPI responses (schedules, flights, history)
# - Better FR24 parsing + UTC→local conversion
# - Cleaner concurrency, caching, logging, and no accidental shadowing of stdlib modules
# - Defensive code against missing keys and shape changes
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

# Cache: (flight_no_upper, date_mmddyyyy) -> (ts, ValidationResult)
_VALIDATION_CACHE: Dict[Tuple[str, str], Tuple[float, "ValidationResult"]] = {}
_CACHE_TTL_S = 15 * 60
_CACHE_MAX = 1000

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
# AEROAPI CLIENT (FlightAware)
# ─────────────────────────────────────────────────────────────────────────────

class AeroAPIClient:
    """
    FlightAware AeroAPI v4 client (read-only) with robust fallbacks.

    Endpoints used:
      - GET /flights/{ident}?start={iso}&end={iso}
      - GET /history/flights/{ident}?start={iso}&end={iso}
      - GET /schedules/{date_start}/{date_end}?ident={DL9013}
        or /schedules/{date_start}/{date_end}?airline={DL}&flight_number={9013}

    Notes:
      • For flights beyond the 2-day live window, /schedules is preferred.
      • We attempt 'ident' filter first; if the API rejects it, we retry airline+flight_number.
      • We always pass query params through aiohttp (no "shadow" locals).
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

    async def __aenter__(self) -> "AeroAPIClient":
        timeout = aiohttp.ClientTimeout(total=12, connect=3)
        self._session = aiohttp.ClientSession(
            headers=self._headers, connector=self._connector, timeout=timeout
        )
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._session:
            await self._session.close()

    async def search(self, flight_no: str, date_mmddyyyy: str) -> Optional[Dict[str, Any]]:
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

        async def _do_get(path: str, params: Dict[str, Any]) -> Tuple[int, Any, float]:
            url = f"{AEROAPI_BASE_URL}{path}"
            t0 = time.perf_counter()
            async with self._session.get(url, params=params) as r:
                elapsed = time.perf_counter() - t0
                status = r.status
                try:
                    body = await r.json()
                except Exception:
                    body = await r.text()
                logger.info(f"AeroAPI GET {path} status={status} took={elapsed:.2f}s")
                return status, body, elapsed

        # Strategy matrix:
        # • Live / near past: /flights/{ident}?start&end
        # • Far past: /history/flights/{ident}?start&end
        # • Far future or schedule-only: /schedules/{start}/{end}?ident=IDENT (or airline+flight_number)
        try_order: List[Tuple[str, Dict[str, Any]]] = []

        if days_delta < -10:
            # Older than ~10 days → use history
            try_order.append((f"/history/flights/{ident}", {"start": day_start_iso, "end": day_end_iso}))
        elif -10 <= days_delta <= 2:
            # within 2 days → flights endpoint can work
            try_order.append((f"/flights/{ident}", {"start": day_start_iso, "end": day_end_iso}))
        else:
            # future (more than ~2 days) → schedules
            # 1) Preferred: schedules with ident=DL9013 (works for many airlines)
            # 2) Fallback: schedules with airline=DL&flight_number=9013
            try_order.append((f"/schedules/{flight_dt.strftime('%Y-%m-%d')}/{(flight_dt + timedelta(days=1)).strftime('%Y-%m-%d')}",
                              {"ident": ident}))
            airline, number, _suffix = _split_ident(ident)
            if number:
                sched_params: Dict[str, Any] = {"flight_number": number}
                if airline:
                    sched_params["airline"] = airline
                try_order.append((f"/schedules/{flight_dt.strftime('%Y-%m-%d')}/{(flight_dt + timedelta(days=1)).strftime('%Y-%m-%d')}",
                                  sched_params))

        # Always also consider cross-trying schedules vs flights if first bucket returns nothing/400.
        # Append alternates to widen the net without being too spammy.
        # (But we won’t duplicate what’s already in try_order.)
        def _have(path_prefix: str) -> bool:
            return any(p.startswith(path_prefix) for p, _ in try_order)

        if not _have("/flights/"):
            try_order.append((f"/flights/{ident}", {"start": day_start_iso, "end": day_end_iso}))
        if not _have("/history/"):
            try_order.append((f"/history/flights/{ident}", {"start": day_start_iso, "end": day_end_iso}))
        if not _have("/schedules/"):
            try_order.append((f"/schedules/{flight_dt.strftime('%Y-%m-%d')}/{(flight_dt + timedelta(days=1)).strftime('%Y-%m-%d')}",
                              {"ident": ident}))

        # Execute sequentially (AeroAPI rate-limits; keep polite)
        last_error_text: Optional[str] = None
        for path, params in try_order:
            status, body, _elapsed = await _do_get(path, params)

            if status == 400:
                # Often indicates wrong param (e.g., 'flight_ident' used in old code)
                # Log first 200 chars to help debugging
                err_txt = body if isinstance(body, str) else json.dumps(body)[:200]
                logger.error(f"AeroAPI 400 on {path} params={params} err={err_txt}")
                last_error_text = err_txt
                continue
            if status == 404:
                continue
            if status != 200:
                last_error_text = body if isinstance(body, str) else json.dumps(body)[:200]
                continue

            # Parse structural variants
            try:
                if not isinstance(body, dict):
                    continue

                # Typical shapes:
                # • flights endpoints: {"flights": [ ... ]}
                # • schedules: {"scheduled": [ ... ]}  (or sometimes "flights" in older variants)
                # • history: {"flights": [ ... ]}

                flights_list: List[Dict[str, Any]] = []

                if "flights" in body and isinstance(body["flights"], list):
                    flights_list = body["flights"]
                elif "scheduled" in body and isinstance(body["scheduled"], list):
                    flights_list = body["scheduled"]
                elif "data" in body and isinstance(body["data"], list):
                    flights_list = body["data"]
                else:
                    # Some responses nest data differently; search for list of dicts
                    for key, val in body.items():
                        if isinstance(val, list) and val and isinstance(val[0], dict):
                            flights_list = val
                            break

                if not flights_list:
                    continue

                # Best-effort pick for target date/ident
                # For schedules we may need to match ident or airline+number ourselves
                def _matches_ident(f: Dict[str, Any]) -> bool:
                    # fields that may contain ident
                    for k in ("ident", "ident_iata", "ident_icao", "flight_number"):
                        v = f.get(k)
                        if isinstance(v, str) and v.strip().upper() == ident:
                            return True
                    # some schedules split (airline, flight_number)
                    airline, number, _sfx = _split_ident(ident)
                    if number:
                        if (str(f.get("flight_number") or "").strip() == number and
                            ((f.get("airline") or f.get("operator") or f.get("operator_iata") or f.get("airline_iata") or "") or airline)):
                            # If we have airline in either the request or item, accept
                            return True
                    return False

                # Prefer exact ident match; else just return first for the day
                for item in flights_list:
                    if _matches_ident(item):
                        return item

                return flights_list[0]  # best we can do

            except Exception as parse_err:
                logger.error(f"AeroAPI parse error: {parse_err}")
                continue

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
            headers={"Accept": "application/json", "User-Agent": "Flight-Intel/2.2"},
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
# VALIDATOR
# ─────────────────────────────────────────────────────────────────────────────

class FastFlightValidator:
    """
    Validates & enriches each extracted flight.
    Fills: origin, dest, sched_out_local, sched_in_local (if missing)
    """

    def __init__(self) -> None:
        self._has_aero = bool(FLIGHTAWARE_API_KEY)
        self._has_fr24 = bool(FLIGHTRADAR24_API_KEY)
        self._aero: Optional[AeroAPIClient] = AeroAPIClient(FLIGHTAWARE_API_KEY) if self._has_aero else None
        self._fr24: Optional[FR24Client] = FR24Client(FLIGHTRADAR24_API_KEY) if self._has_fr24 else None

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

        # cache
        cached = self._cache_get(flight_no, date)
        if cached:
            return cached

        # fast pass if already complete
        if all(flight.get(k) for k in ("origin", "dest", "sched_out_local", "sched_in_local")):
            res = ValidationResult(is_valid=True, confidence=1.0, source="complete")
            self._cache_put(flight_no, date, res)
            return res

        missing = [k for k in ("origin", "dest", "sched_out_local", "sched_in_local") if not flight.get(k)]
        filled: Dict[str, Any] = {}
        source_tags: List[str] = []
        conf = 0.0

        # Execute calls (in parallel, but small fan-out)
        api_results: List[Tuple[str, Optional[Dict[str, Any]]]] = []

        async def _call_aero() -> None:
            if not self._aero:
                api_results.append(("aeroapi", None))
                return
            data = await self._aero.search(flight_no, date)
            api_results.append(("aeroapi", data))

        async def _call_fr24() -> None:
            if not self._fr24:
                api_results.append(("fr24", None))
                return
            data = await self._fr24.search(flight_no, date)
            api_results.append(("fr24", data))

        await asyncio.gather(_call_aero(), _call_fr24())

        # Process results in priority order: AeroAPI first, then FR24
        for tag, data in api_results:
            if not data:
                continue

            if tag == "aeroapi":
                self._apply_aero_fields(data, flight, filled)
                if filled:
                    source_tags.append("aeroapi")
                    conf = max(conf, 0.95)
            elif tag == "fr24":
                self._apply_fr24_fields(data, flight, filled)
                if filled:
                    source_tags.append("fr24")
                    conf = max(conf, 0.85)

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

    def _apply_aero_fields(self, api_data: Dict[str, Any], flight: Dict[str, Any], filled: Dict[str, Any]) -> None:
        """
        Extract origin/dest + scheduled times from AeroAPI record.
        Works with both /flights and /schedules shapes.
        """
        # Origin / Dest
        origin = (
            _normalize_airport_fast(api_data.get("origin"))
            or _normalize_airport_fast(api_data.get("departure_airport"))
        )
        dest = (
            _normalize_airport_fast(api_data.get("destination"))
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
                if isinstance(v, str) and "T" in v:
                    return v
                if isinstance(v, dict):
                    # e.g., {"scheduled": "2025-...Z"}
                    sched = v.get("scheduled")
                    if isinstance(sched, str) and "T" in sched:
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
            sem = asyncio.Semaphore(10)

            async def _lim(vf: Dict[str, Any]) -> ValidationResult:
                async with sem:
                    try:
                        return await self._validate_one(vf)
                    except Exception as e:
                        logger.error(f"Validation exception: {e}")
                        return ValidationResult(is_valid=False, confidence=0.0, source="error", warnings=[str(e)])

            results = await asyncio.gather(*[_lim(f) for f in need], return_exceptions=True)
            for r in results:
                if isinstance(r, Exception):
                    validations.append(ValidationResult(is_valid=False, confidence=0.0, source="error", warnings=[str(r)]))
                else:
                    validations.append(r)

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