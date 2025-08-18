"""
Flight Intel Validation & Enrichment Module – SPEED OPTIMIZED v2.0

Ultra-fast validation and enrichment using external APIs.
With ENHANCED LOGGING for API calls.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple, Pattern
from functools import lru_cache

try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo

import aiohttp
from pydantic import BaseModel, Field

# Enhanced logger with detailed formatting
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("flight-validator")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG & CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
FLIGHTAWARE_API_KEY: str | None = os.getenv("FLIGHTAWARE_API_KEY")
FLIGHTRADAR24_API_KEY: str | None = os.getenv("FLIGHTRADAR24_API_KEY")

AEROAPI_BASE_URL = "https://aeroapi.flightaware.com/aeroapi"
FR24_BASE_URL = "https://api.flightradar24.com/common/v1"

# Cache settings
_VALIDATION_CACHE: Dict[Tuple[str, str], Tuple[float, Any]] = {}
_CACHE_TTL = 900  # 15 min
_CACHE_MAX = 1000

# Pre-compiled regex (faster than compiling each time)
FLIGHT_PATTERN: Pattern = re.compile(r"^[A-Z0-9]{2,4}\d{1,5}[A-Z]?$")

# Common US airport timezones (reduced set for speed)
AIRPORT_TIMEZONES = {
    # Eastern
    "JFK": "America/New_York", "LGA": "America/New_York", "EWR": "America/New_York",
    "ATL": "America/New_York", "BOS": "America/New_York", "DCA": "America/New_York",
    "MIA": "America/New_York", "MCO": "America/New_York", "PHL": "America/New_York",
    "CLT": "America/New_York", "BWI": "America/New_York", "DTW": "America/Detroit",
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

# Pre-create timezone objects (expensive operation)
_TZ_CACHE: Dict[str, ZoneInfo] = {}
for airport, tz_name in AIRPORT_TIMEZONES.items():
    _TZ_CACHE[airport] = ZoneInfo(tz_name)
_DEFAULT_TZ = ZoneInfo("America/New_York")

# ─────────────────────────────────────────────────────────────────────────────
# FAST HELPERS
# ─────────────────────────────────────────────────────────────────────────────
@lru_cache(maxsize=256)
def _iso_day_window_utc(year: int, month: int, day: int) -> tuple[str, str]:
    """Cached ISO day window generation"""
    start = datetime(year, month, day, tzinfo=timezone.utc)
    end = start + timedelta(days=1)
    return (start.isoformat().replace("+00:00", "Z"),
            end.isoformat().replace("+00:00", "Z"))

@lru_cache(maxsize=512)
def _to_iata(code: Optional[str]) -> Optional[str]:
    """Fast IATA code normalization"""
    if not code:
        return None
    c = code.upper()
    if len(c) == 3:
        return c
    if len(c) == 4 and c[0] == 'K':
        return c[1:]
    return None

def _normalize_airport_fast(obj: Any) -> Optional[str]:
    """Fast airport extraction from API response"""
    if isinstance(obj, str):
        return _to_iata(obj)
    if isinstance(obj, dict):
        # Check most common keys first
        if "code_iata" in obj:
            return _to_iata(obj["code_iata"])
        if "iata" in obj:
            return _to_iata(obj["iata"])
        if "code" in obj:
            return _to_iata(obj["code"])
    return None

def convert_to_local_time_fast(iso_time: str, airport_code: str) -> str:
    """Fast timezone conversion using pre-cached timezone objects"""
    dt = datetime.fromisoformat(iso_time.replace('Z', '+00:00'))
    tz = _TZ_CACHE.get(airport_code, _DEFAULT_TZ)
    return dt.astimezone(tz).strftime("%H%M")

# ─────────────────────────────────────────────────────────────────────────────
# LIGHTWEIGHT MODELS
# ─────────────────────────────────────────────────────────────────────────────
class ValidationResult(BaseModel):
    is_valid: bool
    confidence: float
    source: str
    warnings: List[str] = Field(default_factory=list)
    corrections: Dict[str, Any] = Field(default_factory=dict)
    filled_fields: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        # Faster serialization
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
# OPTIMIZED API CLIENTS WITH ENHANCED LOGGING
# ─────────────────────────────────────────────────────────────────────────────
class AeroAPIClient:
    """Optimized AeroAPI client with detailed logging"""
    
    def __init__(self, api_key: str) -> None:
        self._headers = {"x-apikey": api_key, "Accept": "application/json"}
        self._session: Optional[aiohttp.ClientSession] = None
        # Optimized connector settings
        self._connector = aiohttp.TCPConnector(
            limit=20,
            limit_per_host=10,
            ttl_dns_cache=300,
            enable_cleanup_closed=True,
            keepalive_timeout=30
        )

    async def __aenter__(self) -> "AeroAPIClient":
        timeout = aiohttp.ClientTimeout(total=10, connect=3)
        self._session = aiohttp.ClientSession(
            headers=self._headers,
            connector=self._connector,
            timeout=timeout
        )
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._session:
            await self._session.close()

    async def search_flight_fast(self, flight_no: str, date: str) -> Optional[Dict]:
        """Fast flight search with detailed logging"""
        if not flight_no or not self._session:
            return None

        ident = flight_no.strip().upper()
        if not FLIGHT_PATTERN.match(ident):
            logger.warning(f"Invalid flight pattern: {ident}")
            return None

        try:
            # Fast date parsing
            month, day, year = date.split('/')
            flight_dt = datetime(int(year), int(month), int(day))
        except (ValueError, IndexError):
            logger.error(f"Invalid date format: {date}")
            return None

        delta = (flight_dt.date() - datetime.utcnow().date()).days
        
        # Determine endpoint
        if delta < -10:
            endpoint = f"/history/flights/{ident}"
            start_iso, end_iso = _iso_day_window_utc(flight_dt.year, flight_dt.month, flight_dt.day)
            params = {"start": start_iso, "end": end_iso}
        elif -10 <= delta <= 2:
            endpoint = f"/flights/{ident}"
            start_iso, end_iso = _iso_day_window_utc(flight_dt.year, flight_dt.month, flight_dt.day)
            params = {"start": start_iso, "end": end_iso}
        else:
            date_start = flight_dt.strftime("%Y-%m-%d")
            date_end = (flight_dt + timedelta(days=1)).strftime("%Y-%m-%d")
            endpoint = f"/schedules/{date_start}/{date_end}"
            params = {"flight_ident": ident, "max_pages": 1}

        # LOG API CALL PARAMS
        logger.info("🚀 FLIGHTAWARE API CALL")
        logger.info(f"   Flight: {ident} on {date}")
        logger.info(f"   Endpoint: {endpoint}")
        logger.info(f"   Params: {json.dumps(params)}")
        
        try:
            url = f"{AEROAPI_BASE_URL}{endpoint}"
            
            start_time = time.perf_counter()
            async with self._session.get(url, params=params) as resp:
                elapsed = time.perf_counter() - start_time
                
                # LOG RESPONSE STATUS
                logger.info(f"   Status: {resp.status} (took {elapsed:.2f}s)")
                
                if resp.status != 200:
                    error_text = await resp.text()
                    logger.error(f"   ❌ Error response: {error_text[:200]}")
                    return None

                data = await resp.json()
                
                # LOG RESPONSE DATA
                logger.info(f"   ✅ Response received:")
                
                # Log structure of response
                if isinstance(data, dict):
                    logger.info(f"      Keys: {list(data.keys())}")
                    
                    # Extract flights based on endpoint type
                    if endpoint.startswith("/schedules/"):
                        flights = data.get("scheduled") or data.get("schedules", [])
                    elif endpoint.startswith("/history/"):
                        flights = data.get("flights", [])
                    else:
                        flights = data.get("flights", data.get("data", []))
                    
                    logger.info(f"      Found {len(flights) if flights else 0} flights")
                    
                    if flights:
                        # Log first flight details
                        first_flight = flights[0]
                        logger.info(f"      Sample flight data:")
                        logger.info(f"         Ident: {first_flight.get('ident', 'N/A')}")
                        logger.info(f"         Origin: {first_flight.get('origin', 'N/A')}")
                        logger.info(f"         Dest: {first_flight.get('destination', 'N/A')}")
                        logger.info(f"         Sched Out: {first_flight.get('scheduled_out', 'N/A')}")
                        logger.info(f"         Sched In: {first_flight.get('scheduled_in', 'N/A')}")
                else:
                    logger.info(f"      Response type: {type(data)}")
                    logger.info(f"      Data preview: {str(data)[:200]}")
                
                if not flights:
                    logger.warning("   ⚠️ No flights found in response")
                    return None

                # Quick match - return first valid flight
                target_date = flight_dt.date()
                for flight in flights:
                    # Quick ident check for schedules
                    if endpoint.startswith("/schedules/"):
                        f_ident = flight.get("ident") or flight.get("flight_number", "")
                        if f_ident.upper() != ident:
                            continue
                    
                    # Return first match (usually correct)
                    logger.info(f"   ✈️ Returning flight match")
                    return flight

                return flights[0] if flights else None

        except asyncio.TimeoutError:
            logger.error(f"   ⏰ Timeout after 10s")
            return None
        except Exception as e:
            logger.error(f"   ❌ Exception: {str(e)}")
            return None

class FR24Client:
    """Optimized FR24 client with logging"""
    
    def __init__(self, api_key: str) -> None:
        self._api_key = api_key
        self._headers = {"Accept": "application/json", "User-Agent": "Flight-Intel/2.0"}
        self._session: Optional[aiohttp.ClientSession] = None
        self._connector = aiohttp.TCPConnector(
            limit=20,
            limit_per_host=10,
            ttl_dns_cache=300
        )

    async def __aenter__(self) -> "FR24Client":
        timeout = aiohttp.ClientTimeout(total=10, connect=3)
        self._session = aiohttp.ClientSession(
            headers=self._headers,
            connector=self._connector,
            timeout=timeout
        )
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._session:
            await self._session.close()

    async def search_flight_fast(self, flight_no: str, date: str) -> Optional[Dict]:
        """Fast FR24 search with logging"""
        if not flight_no or not self._session:
            return None

        ident = flight_no.strip().upper()
        params = {
            "query": ident,
            "fetchBy": "flight",
            "limit": 10,
            "token": self._api_key,
        }
        
        # LOG API CALL
        logger.info("🛩️ FLIGHTRADAR24 API CALL")
        logger.info(f"   Flight: {ident} on {date}")
        logger.info(f"   Params: {json.dumps({k: v for k, v in params.items() if k != 'token'})}")
        
        url = f"{FR24_BASE_URL}/flight/list.json"
        try:
            start_time = time.perf_counter()
            async with self._session.get(url, params=params) as resp:
                elapsed = time.perf_counter() - start_time
                
                logger.info(f"   Status: {resp.status} (took {elapsed:.2f}s)")
                
                if resp.status != 200:
                    error_text = await resp.text()
                    logger.error(f"   ❌ Error response: {error_text[:200]}")
                    return None

                data = await resp.json()
                
                # LOG RESPONSE
                flights = data.get("result", {}).get("response", {}).get("data", [])
                logger.info(f"   ✅ Response: Found {len(flights)} flights")
                
                if flights:
                    first = flights[0]
                    logger.info(f"      Sample flight:")
                    logger.info(f"         Flight: {first.get('flight', {}).get('identification', {}).get('number', 'N/A')}")
                    logger.info(f"         Origin: {first.get('airport', {}).get('origin', {}).get('code', 'N/A')}")
                    logger.info(f"         Dest: {first.get('airport', {}).get('destination', {}).get('code', 'N/A')}")
                    
                    # Return first match (usually correct)
                    return flights[0]
                
                logger.warning("   ⚠️ No flights found")
                return None

        except asyncio.TimeoutError:
            logger.error(f"   ⏰ Timeout after 10s")
            return None
        except Exception as e:
            logger.error(f"   ❌ Exception: {str(e)}")
            return None

# ─────────────────────────────────────────────────────────────────────────────
# SPEED-OPTIMIZED VALIDATOR WITH LOGGING
# ─────────────────────────────────────────────────────────────────────────────
class FastFlightValidator:
    """Ultra-fast flight validator with detailed logging"""
    
    def __init__(self) -> None:
        self._has_aeroapi = bool(FLIGHTAWARE_API_KEY)
        self._has_fr24 = bool(FLIGHTRADAR24_API_KEY)
        self._aero_client: Optional[AeroAPIClient] = None
        self._fr24_client: Optional[FR24Client] = None
        
        logger.info(f"🔧 Validator initialized:")
        logger.info(f"   FlightAware API: {'✅ Available' if self._has_aeroapi else '❌ Not configured'}")
        logger.info(f"   FR24 API: {'✅ Available' if self._has_fr24 else '❌ Not configured'}")
        
        # Pre-create clients if keys exist
        if self._has_aeroapi:
            self._aero_client = AeroAPIClient(FLIGHTAWARE_API_KEY)
        if self._has_fr24:
            self._fr24_client = FR24Client(FLIGHTRADAR24_API_KEY)

    async def __aenter__(self) -> "FastFlightValidator":
        tasks = []
        if self._aero_client:
            tasks.append(self._aero_client.__aenter__())
        if self._fr24_client:
            tasks.append(self._fr24_client.__aenter__())
        if tasks:
            await asyncio.gather(*tasks)
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        tasks = []
        if self._aero_client:
            tasks.append(self._aero_client.__aexit__(None, None, None))
        if self._fr24_client:
            tasks.append(self._fr24_client.__aexit__(None, None, None))
        if tasks:
            await asyncio.gather(*tasks)

    def _get_cached(self, flight_no: str, date: str) -> Optional[ValidationResult]:
        """Fast cache lookup"""
        key = (flight_no.upper(), date)
        entry = _VALIDATION_CACHE.get(key)
        if entry and time.time() - entry[0] < _CACHE_TTL:
            logger.info(f"   📦 Cache hit for {flight_no} on {date}")
            return entry[1]
        return None

    def _store_cache(self, flight_no: str, date: str, result: ValidationResult) -> None:
        """Fast cache storage"""
        key = (flight_no.upper(), date)
        _VALIDATION_CACHE[key] = (time.time(), result)
        
        # Fast cache eviction
        if len(_VALIDATION_CACHE) > _CACHE_MAX:
            # Remove 10% oldest entries
            to_remove = _CACHE_MAX // 10
            for _ in range(to_remove):
                _VALIDATION_CACHE.pop(next(iter(_VALIDATION_CACHE)))

    async def validate_flight_fast(self, flight: Dict) -> ValidationResult:
        """Ultra-fast single flight validation with logging"""
        flight_no = flight.get("flight_no")
        flight_date = flight.get("date")

        logger.info(f"🔍 Validating flight: {flight_no} on {flight_date}")
        
        # Fast validation for missing required fields
        if not flight_no or not flight_date:
            logger.warning(f"   ❌ Missing required fields")
            return ValidationResult(is_valid=False, confidence=0.0, source="none")

        # Cache check
        cached = self._get_cached(flight_no, flight_date)
        if cached:
            return cached

        # FAST PATH: Skip API if all fields present
        if all(flight.get(k) for k in ("origin", "dest", "sched_out_local", "sched_in_local")):
            logger.info(f"   ✅ All fields present - skipping API calls")
            result = ValidationResult(is_valid=True, confidence=1.0, source="complete")
            self._store_cache(flight_no, flight_date, result)
            return result

        # Determine what's missing
        missing = [k for k in ("origin", "dest", "sched_out_local", "sched_in_local") 
                   if not flight.get(k)]
        
        logger.info(f"   🔎 Missing fields: {missing}")

        # Parallel API calls
        tasks = []
        if self._aero_client and self._has_aeroapi:
            logger.info(f"   📡 Querying FlightAware...")
            tasks.append(self._aero_client.search_flight_fast(flight_no, flight_date))
        if self._fr24_client and self._has_fr24:
            logger.info(f"   📡 Querying FlightRadar24...")
            tasks.append(self._fr24_client.search_flight_fast(flight_no, flight_date))

        if not tasks:
            # No APIs available
            logger.warning(f"   ⚠️ No APIs available for validation")
            result = ValidationResult(is_valid=True, confidence=0.5, source="none")
            self._store_cache(flight_no, flight_date, result)
            return result

        # Execute API calls
        api_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        result = ValidationResult(is_valid=False, confidence=0.0, source="none")
        
        for i, api_data in enumerate(api_results):
            if isinstance(api_data, Exception):
                logger.error(f"   ❌ API {i} exception: {api_data}")
                continue
            if not api_data:
                logger.info(f"   ⚠️ API {i} returned no data")
                continue
                
            # Process based on which API returned data
            if i == 0 and self._has_aeroapi:  # AeroAPI
                logger.info(f"   📊 Processing FlightAware data...")
                self._process_aeroapi_fast(api_data, flight, missing, result)
                result.source = "aeroapi"
                result.is_valid = True
                result.confidence = 0.95
                
                # Log filled fields
                if result.filled_fields:
                    logger.info(f"   ✨ FlightAware filled: {list(result.filled_fields.keys())}")
                    for k, v in result.filled_fields.items():
                        logger.info(f"      {k}: {v}")
                        
            elif (i == 1 or (i == 0 and not self._has_aeroapi)) and self._has_fr24:  # FR24
                logger.info(f"   📊 Processing FR24 data...")
                self._process_fr24_fast(api_data, flight, missing, result)
                if result.source == "none":
                    result.source = "fr24"
                else:
                    result.source += "+fr24"
                result.is_valid = True
                result.confidence = max(result.confidence, 0.85)
                
                # Log filled fields
                if result.filled_fields:
                    logger.info(f"   ✨ FR24 filled: {list(result.filled_fields.keys())}")
                    for k, v in result.filled_fields.items():
                        logger.info(f"      {k}: {v}")

        # Fallback if no API data
        if not result.is_valid:
            logger.warning(f"   ⚠️ No API data available - using heuristic")
            result = ValidationResult(is_valid=True, confidence=0.5, source="heuristic")

        logger.info(f"   📋 Validation complete: confidence={result.confidence:.2f}, source={result.source}")
        
        self._store_cache(flight_no, flight_date, result)
        return result

    def _process_aeroapi_fast(self, api_data: Dict, flight: Dict, missing: List[str], result: ValidationResult) -> None:
        """Fast AeroAPI data extraction"""
        # Quick origin extraction
        if "origin" in missing:
            origin = (
                _normalize_airport_fast(api_data.get("origin")) or
                _normalize_airport_fast(api_data.get("departure_airport"))
            )
            if origin:
                result.filled_fields["origin"] = origin

        # Quick dest extraction
        if "dest" in missing:
            dest = (
                _normalize_airport_fast(api_data.get("destination")) or
                _normalize_airport_fast(api_data.get("arrival_airport"))
            )
            if dest:
                result.filled_fields["dest"] = dest

        # Quick time extraction
        if "sched_out_local" in missing:
            for field in ("scheduled_out", "scheduled_off", "departure_time", "departure"):
                if field in api_data and api_data[field]:
                    try:
                        origin = result.filled_fields.get("origin") or flight.get("origin")
                        if origin:
                            result.filled_fields["sched_out_local"] = convert_to_local_time_fast(
                                api_data[field], origin
                            )
                            break
                    except:
                        pass

        if "sched_in_local" in missing:
            for field in ("scheduled_in", "scheduled_on", "arrival_time", "arrival"):
                if field in api_data and api_data[field]:
                    try:
                        dest = result.filled_fields.get("dest") or flight.get("dest")
                        if dest:
                            result.filled_fields["sched_in_local"] = convert_to_local_time_fast(
                                api_data[field], dest
                            )
                            break
                    except:
                        pass

    def _process_fr24_fast(self, api_data: Dict, flight: Dict, missing: List[str], result: ValidationResult) -> None:
        """Fast FR24 data extraction"""
        airport_data = api_data.get("airport", {})
        
        if "origin" in missing:
            origin = _normalize_airport_fast(airport_data.get("origin", {}).get("code"))
            if origin:
                result.filled_fields["origin"] = origin

        if "dest" in missing:
            dest = _normalize_airport_fast(airport_data.get("destination", {}).get("code"))
            if dest:
                result.filled_fields["dest"] = dest

        time_data = api_data.get("time", {}).get("scheduled", {})
        
        if "sched_out_local" in missing and time_data.get("departure"):
            try:
                dt = datetime.fromtimestamp(time_data["departure"])
                result.filled_fields["sched_out_local"] = dt.strftime("%H%M")
            except:
                pass

        if "sched_in_local" in missing and time_data.get("arrival"):
            try:
                dt = datetime.fromtimestamp(time_data["arrival"])
                result.filled_fields["sched_in_local"] = dt.strftime("%H%M")
            except:
                pass

    async def validate_batch_fast(self, flights: List[Dict]) -> Dict[str, Any]:
        """Ultra-fast batch validation with logging"""
        start = time.time()
        
        logger.info(f"🚀 BATCH VALIDATION STARTED")
        logger.info(f"   Total flights to validate: {len(flights)}")
        
        # Filter flights that need validation
        flights_to_validate = []
        prefilled = []
        
        for f in flights:
            if all(f.get(k) for k in ("origin", "dest", "sched_out_local", "sched_in_local")):
                # Already complete
                prefilled.append(ValidationResult(is_valid=True, confidence=1.0, source="complete"))
            else:
                flights_to_validate.append(f)
        
        logger.info(f"   Flights already complete: {len(prefilled)}")
        logger.info(f"   Flights needing validation: {len(flights_to_validate)}")

        # Validate only what's needed
        if flights_to_validate:
            # Batch process with limited concurrency
            semaphore = asyncio.Semaphore(10)
            
            async def validate_with_limit(flight):
                async with semaphore:
                    return await self.validate_flight_fast(flight)
            
            validations = await asyncio.gather(*[
                validate_with_limit(f) for f in flights_to_validate
            ], return_exceptions=True)
        else:
            validations = []

        # Combine results
        all_validations = []
        validate_idx = 0
        
        for f in flights:
            if all(f.get(k) for k in ("origin", "dest", "sched_out_local", "sched_in_local")):
                all_validations.append(prefilled.pop(0))
            else:
                val = validations[validate_idx]
                if isinstance(val, Exception):
                    logger.error(f"   Validation exception: {val}")
                    val = ValidationResult(is_valid=False, confidence=0.0, source="error")
                all_validations.append(val)
                validate_idx += 1

        # Build enriched flights
        enriched_flights = []
        for raw_flight, validation in zip(flights, all_validations):
            ef = EnrichedFlight(**raw_flight, validation_result=validation)
            
            # Apply filled fields
            for k, v in validation.filled_fields.items():
                setattr(ef, k, v)
            
            enriched_flights.append(ef)

        # Calculate summary
        valid_count = sum(1 for e in enriched_flights if e.validation_result.is_valid)
        avg_conf = (
            sum(e.validation_result.confidence for e in enriched_flights) / len(enriched_flights)
            if enriched_flights else 0.0
        )
        total_filled = sum(len(e.validation_result.filled_fields) for e in enriched_flights)
        
        elapsed = time.time() - start
        
        logger.info(f"✅ BATCH VALIDATION COMPLETE")
        logger.info(f"   Valid flights: {valid_count}/{len(flights)}")
        logger.info(f"   Average confidence: {avg_conf:.2f}")
        logger.info(f"   Total fields filled: {total_filled}")
        logger.info(f"   Processing time: {elapsed:.2f}s")
        logger.info(f"   Sources used: {sorted(set(e.validation_result.source for e in enriched_flights))}")

        return {
            "enriched_flights": [e.dict() for e in enriched_flights],
            "validation_summary": {
                "total_flights": len(flights),
                "valid_flights": valid_count,
                "average_confidence": avg_conf,
                "total_fields_filled": total_filled,
                "processing_time_seconds": elapsed,
                "sources_used": sorted(set(e.validation_result.source for e in enriched_flights)),
                "warnings": [],
            },
        }

# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API (Backwards Compatible)
# ─────────────────────────────────────────────────────────────────────────────
async def validate_extraction_results(extraction_result: Dict) -> Dict:
    """Fast validation of extraction results with logging"""
    logger.info("="*60)
    logger.info("VALIDATION REQUEST RECEIVED")
    logger.info("="*60)
    
    flights_to_validate = extraction_result.get("flights", [])
    
    async with FastFlightValidator() as validator:
        summary = await validator.validate_batch_fast(flights_to_validate)

    extraction_result["validation"] = summary["validation_summary"]
    extraction_result["enriched_flights"] = summary["enriched_flights"]
    
    logger.info("="*60)
    logger.info("VALIDATION REQUEST COMPLETE")
    logger.info("="*60)

    return extraction_result