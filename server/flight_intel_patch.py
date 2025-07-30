"""
Flight Intel Validation & Firehose Integration Module  – FIXED VERSION

Validates and enriches extracted flight data using external APIs:
- FlightAware AeroAPI v4
- FlightRadar24 API
- FlightAware Firehose streaming data

Key fixes:
- Fixed AeroAPI /schedules endpoint handling
- Proper timezone conversion for scheduled times
- Better field extraction from API responses
- Enhanced error handling and logging
- Removed enriched_data population
- Skip validation when all fields present
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

try:
    from zoneinfo import ZoneInfo
except ImportError:
    # Python < 3.9
    from backports.zoneinfo import ZoneInfo

import aiohttp
from pydantic import BaseModel, Field

# Local dependency
from firehose_client import validate_with_firehose

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# ░ CONFIGURATION ░
# ─────────────────────────────────────────────────────────────────────────────
FLIGHTAWARE_API_KEY: str | None = os.getenv("FLIGHTAWARE_API_KEY")
FLIGHTRADAR24_API_KEY: str | None = os.getenv("FLIGHTRADAR24_API_KEY")
FIREHOSE_USERNAME: str | None = os.getenv("FIREHOSE_USERNAME")
FIREHOSE_PASSWORD: str | None = os.getenv("FIREHOSE_PASSWORD")

AEROAPI_BASE_URL = "https://aeroapi.flightaware.com/aeroapi"
FR24_BASE_URL = "https://api.flightradar24.com/common/v1"

# Airport timezone mapping for common US airports
AIRPORT_TIMEZONES = {
    # Eastern Time
    "JFK": "America/New_York", "LGA": "America/New_York", "EWR": "America/New_York",
    "ATL": "America/New_York", "BOS": "America/New_York", "DCA": "America/New_York",
    "MIA": "America/New_York", "MCO": "America/New_York", "PHL": "America/New_York",
    "CLT": "America/New_York", "BWI": "America/New_York", "FLL": "America/New_York",
    "TPA": "America/New_York", "RDU": "America/New_York", "PIT": "America/New_York",
    
    # Central Time
    "ORD": "America/Chicago", "MDW": "America/Chicago", "DFW": "America/Chicago",
    "IAH": "America/Chicago", "MSP": "America/Chicago", "STL": "America/Chicago",
    "MCI": "America/Chicago", "MKE": "America/Chicago", "MSY": "America/Chicago",
    "BNA": "America/Chicago", "AUS": "America/Chicago", "SAT": "America/Chicago",
    "MEM": "America/Chicago", "OKC": "America/Chicago",
    
    # Mountain Time  
    "DEN": "America/Denver", "SLC": "America/Denver", "ABQ": "America/Denver",
    
    # Pacific Time
    "LAX": "America/Los_Angeles", "SFO": "America/Los_Angeles", "SEA": "America/Los_Angeles",
    "SAN": "America/Los_Angeles", "PDX": "America/Los_Angeles", "LAS": "America/Los_Angeles",
    "SJC": "America/Los_Angeles", "OAK": "America/Los_Angeles", "SMF": "America/Los_Angeles",
    "BUR": "America/Los_Angeles", "ONT": "America/Los_Angeles", "SNA": "America/Los_Angeles",
    
    # Arizona (no DST)
    "PHX": "America/Phoenix", "TUS": "America/Phoenix",
    
    # Alaska
    "ANC": "America/Anchorage", "FAI": "America/Anchorage", "JNU": "America/Anchorage",
    
    # Hawaii
    "HNL": "Pacific/Honolulu", "OGG": "Pacific/Honolulu", "KOA": "Pacific/Honolulu",
    "LIH": "Pacific/Honolulu", "ITO": "Pacific/Honolulu",
    
    # Detroit (Eastern but sometimes listed separately)
    "DTW": "America/Detroit",
    
    # Puerto Rico (Atlantic)
    "SJU": "America/Puerto_Rico",
}

# ─────────────────────────────────────────────────────────────────────────────
# ░ HELPER ░
# ─────────────────────────────────────────────────────────────────────────────
def choose_validation_path(date_str: str) -> str:
    """
    Decide which validator to use based on date.
    Returns 'schedule' if the flight date is strictly more than 2 days in the future,
    otherwise 'live'.
    """
    try:
        flight_dt = datetime.strptime(date_str, "%m/%d/%Y").replace(tzinfo=timezone.utc)
    except ValueError:
        return "schedule"

    delta_days = (flight_dt.date() - datetime.utcnow().date()).days
    return "schedule" if delta_days > 2 else "live"

def convert_to_local_time(iso_time: str, airport_code: str) -> str:
    dt = datetime.fromisoformat(iso_time.replace('Z', '+00:00'))
    code = _to_iata(airport_code) or airport_code
    tz_name = AIRPORT_TIMEZONES.get(code, "America/New_York")
    return dt.astimezone(ZoneInfo(tz_name)).strftime("%H%M")


def _iso_day_window_utc(day: datetime) -> tuple[str, str]:
    start = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
    end   = start + timedelta(days=1)
    # ensure trailing 'Z'
    return (
        start.isoformat().replace("+00:00", "Z"),
        end.isoformat().replace("+00:00", "Z"),
    )

def _to_iata(code: Optional[str]) -> Optional[str]:
    if not code:
        return None
    c = code.upper()
    if len(c) == 3:               # already IATA
        return c
    if len(c) == 4 and c.startswith("K"):  # US ICAO → IATA
        return c[1:]
    return None

def _normalize_airport(obj: Any) -> Optional[str]:
    # obj may be a dict from AeroAPI (with code, code_iata, code_icao) or a string
    if isinstance(obj, dict):
        for k in ("code_iata", "iata", "iata_code"):
            v = _to_iata(obj.get(k))
            if v: return v
        for k in ("code", "code_icao", "icao"):
            v = _to_iata(obj.get(k))
            if v: return v
        return None
    if isinstance(obj, str):
        return _to_iata(obj)
    return None

# ─────────────────────────────────────────────────────────────────────────────
# ░ DATA MODELS ░
# ─────────────────────────────────────────────────────────────────────────────
class ValidationResult(BaseModel):
    """Outcome of a single‑flight validation attempt."""
    is_valid: bool = Field(..., description="Whether the flight is considered valid")
    confidence: float = Field(..., description="Confidence score (0‑1)")
    source: str = Field(..., description="Which source(s) validated the flight")
    warnings: List[str] = Field(default_factory=list)
    corrections: Dict[str, Any] = Field(default_factory=dict)
    enriched_data: Dict[str, Any] = Field(default_factory=dict)  # Keep for backward compatibility but don't use
    filled_fields: Dict[str, Any] = Field(default_factory=dict, description="Fields that were missing and now filled")


class EnrichedFlight(BaseModel):
    """A flight + any corrections / enrichment returned by validators."""
    date: str
    flight_no: str
    origin: Optional[str] = None
    dest: Optional[str] = None
    sched_out_local: Optional[str] = None
    sched_in_local: Optional[str] = None
    actual_out: Optional[str] = None
    actual_in: Optional[str] = None
    aircraft_type: Optional[str] = None
    aircraft_reg: Optional[str] = None
    flight_status: Optional[str] = None
    validation_result: Optional[ValidationResult] = None
    api_data: Dict[str, Any] = Field(default_factory=dict)

# ─────────────────────────────────────────────────────────────────────────────
# ░ API CLIENTS ░
# ─────────────────────────────────────────────────────────────────────────────
class AeroAPIClient:
    """Enhanced wrapper around FlightAware AeroAPI."""

    def __init__(self, api_key: str) -> None:
        self._headers = {"x-apikey": api_key, "Accept": "application/json"}
        self._session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self) -> "AeroAPIClient":
        self._session = aiohttp.ClientSession(headers=self._headers)
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._session:
            await self._session.close()

    async def search_flight(self, flight_no: str, date: str) -> Optional[Dict]:
        """Search for flight data from AeroAPI with proper endpoint selection."""
        if not flight_no:
            return None

        ident = flight_no.strip().upper()
        if not re.match(r"^[A-Z0-9]{2,4}\d{1,5}[A-Z]?$", ident):
            logger.debug("AeroAPI ident rejected: %s", ident)
            return None

        try:
            flight_dt = datetime.strptime(date, "%m/%d/%Y")
        except ValueError:
            logger.debug("Invalid date for AeroAPI search: %s", date)
            return None

        delta = (flight_dt.date() - datetime.utcnow().date()).days
        
        # Log the decision making
        logger.info(f"AeroAPI search for {ident} on {date} (delta: {delta} days)")
        
        # Historical flights (more than 10 days ago)
        if delta < -10:
            endpoint = f"/history/flights/{ident}"
            start_iso, end_iso = _iso_day_window_utc(flight_dt)
            params = {"start": start_iso, "end": end_iso}
        # Recent past and near future (within 2 days)
        elif -10 <= delta <= 2:
            endpoint = f"/flights/{ident}"
            start_iso, end_iso = _iso_day_window_utc(flight_dt)
            params = {"start": start_iso, "end": end_iso}
        # Future scheduled flights (more than 2 days out)
        else:
            # For future schedules, we need to use the general schedules endpoint
            # and filter by flight_ident
            date_start = flight_dt.strftime("%Y-%m-%d")
            date_end = (flight_dt + timedelta(days=1)).strftime("%Y-%m-%d")
            endpoint = f"/schedules/{date_start}/{date_end}"
            params = {
                "flight_ident": ident,  # Filter for specific flight
                "max_pages": 1
            }

        try:
            url = f"{AEROAPI_BASE_URL}{endpoint}"
            logger.info(f"AeroAPI request: {url} with params: {params}")
            
            async with self._session.get(url, params=params) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    logger.error(f"AeroAPI {ident} returned {resp.status}: {error_text}")
                    return None
                
                data = await resp.json()
                logger.debug(f"AeroAPI raw response: {json.dumps(data, indent=2)[:500]}...")
                
                # Handle different response structures
                flights = []
                
                # For /schedules/{date}/{date} endpoint (future flights)
                if endpoint.startswith("/schedules/") and "/" in endpoint[11:]:
                    # This endpoint returns a different structure
                    if "schedules" in data:
                        flights = data["schedules"]
                    elif isinstance(data, list):
                        flights = data
                    else:
                        flights = []
                # For /history endpoints
                elif endpoint.startswith("/history/"):
                    flights = data.get("flights", [])
                # For standard /flights/{ident} endpoint
                else:
                    flights = data.get("flights", data.get("data", []))
                
                if not flights:
                    logger.info(f"No flights found for {ident} on {date}")
                    return None
                
                logger.info(f"Found {len(flights)} flights for {ident}")
                
                # Find the best match by departure time
                target_date = flight_dt.date()
                best_match = None
                
                # For schedules endpoint, we need to filter by our specific flight
                if endpoint.startswith("/schedules/") and "/" in endpoint[11:]:
                    # The schedules endpoint returns all flights, we need to find ours
                    for flight_data in flights:
                        # Check if this is our flight
                        flight_ident = (
                            flight_data.get("ident") or 
                            flight_data.get("flight_number") or
                            flight_data.get("flight_ident") or
                            ""
                        )
                        
                        if flight_ident.upper() == ident:
                            # Check date
                            date_fields = [
                                "departure_time", "departure", "scheduled_departure_time",
                                "arrival_time", "arrival", "scheduled_arrival_time"
                            ]
                            
                            for field in date_fields:
                                if field in flight_data and flight_data[field]:
                                    try:
                                        departure_dt = datetime.fromisoformat(
                                            flight_data[field].replace('Z', '+00:00')
                                        )
                                        if departure_dt.date() == target_date:
                                            logger.info(f"Found matching scheduled flight on {target_date}")
                                            return flight_data
                                        elif not best_match:
                                            best_match = flight_data
                                    except Exception as e:
                                        logger.debug(f"Error parsing date field {field}: {e}")
                
                else:
                    # For other endpoints, use existing logic
                    for flight in flights:
                        # Check various date fields
                        date_fields = [
                            "scheduled_out", "scheduled_off", "scheduled_departure_time",
                            "actual_out", "actual_off", "actual_departure_time",
                            "filed_departure_time", "estimated_departure_time"
                        ]
                        
                        for field in date_fields:
                            if field in flight and flight[field]:
                                try:
                                    departure_dt = datetime.fromisoformat(
                                        flight[field].replace('Z', '+00:00')
                                    )
                                    if departure_dt.date() == target_date:
                                        logger.info(f"Found matching flight on {target_date}")
                                        return flight
                                    elif not best_match:
                                        best_match = flight
                                except Exception as e:
                                    logger.debug(f"Error parsing date field {field}: {e}")
                
                # Return best match if no exact date match
                if best_match:
                    logger.info(f"Returning best match for {ident}")
                    return best_match
                    
                # Return first result as fallback
                return flights[0] if flights else None
                
        except Exception as e:
            logger.error(f"AeroAPI request error: {e}")
            return None


class FR24Client:
    """Enhanced wrapper for FlightRadar24 API."""

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key
        self._headers = {
            "Accept": "application/json",
            "User-Agent": "Mozilla/5.0 (Flight‑Intel)",
        }
        self._session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self) -> "FR24Client":
        self._session = aiohttp.ClientSession(headers=self._headers)
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._session:
            await self._session.close()

    async def search_flight(self, flight_no: str, date: str) -> Optional[Dict]:
        """Search for flight in FR24 with enhanced date matching."""
        if not flight_no:
            return None

        ident = flight_no.strip().upper()
        
        # Try both flight search and schedule search
        for search_type in ["flight", "schedule"]:
            params = {
                "query": ident,
                "fetchBy": search_type,
                "limit": 25,
                "token": self._api_key,
            }
            url = f"{FR24_BASE_URL}/flight/list.json"

            try:
                async with self._session.get(url, params=params) as resp:
                    if resp.status != 200:
                        logger.debug(f"FR24 {ident} returned {resp.status} for {search_type} search")
                        continue
                    
                    data = await resp.json()
                    flights: List[Dict] = (
                        data.get("result", {}).get("response", {}).get("data", [])
                    )
                    
                    if not flights:
                        continue

                    target_date = datetime.strptime(date, "%m/%d/%Y").date()
                    
                    # Find best match by date
                    for item in flights:
                        # Check scheduled departure
                        ts = item.get("time", {}).get("scheduled", {}).get("departure")
                        if ts:
                            try:
                                flight_date = datetime.fromtimestamp(ts).date()
                                if flight_date == target_date:
                                    return item
                            except:
                                pass
                    
                    # If no exact match but we have results, check if any are close
                    if search_type == "schedule":
                        for item in flights:
                            # For schedules, also check the flight number pattern
                            if item.get("identification", {}).get("number", {}).get("default") == ident:
                                return item
                
            except Exception as exc:
                logger.debug(f"FR24 {search_type} search exception for {ident}: {exc}")
                
        return None


# ─────────────────────────────────────────────────────────────────────────────
# ░ VALIDATOR ░
# ─────────────────────────────────────────────────────────────────────────────
class FlightValidator:
    """Enhanced validator that actively fills missing fields."""

    def __init__(self) -> None:
        self._has_aeroapi = bool(FLIGHTAWARE_API_KEY)
        self._has_fr24 = bool(FLIGHTRADAR24_API_KEY)
        if not self._has_aeroapi:
            logger.warning("AeroAPI key not configured – skipping that source")
        if not self._has_fr24:
            logger.warning("FR24 key not configured – skipping that source")
        if not (FIREHOSE_USERNAME and FIREHOSE_PASSWORD):
            logger.warning("Firehose credentials not configured – Firehose enrichment disabled")

        # Reusable API clients for batch validation
        self._aero_client: Optional[AeroAPIClient] = None
        self._fr24_client: Optional[FR24Client] = None

    async def open_clients(self) -> None:
        """Initialize API clients for reuse across many flights."""
        if self._has_aeroapi and not self._aero_client:
            self._aero_client = AeroAPIClient(FLIGHTAWARE_API_KEY)
            await self._aero_client.__aenter__()
        if self._has_fr24 and not self._fr24_client:
            self._fr24_client = FR24Client(FLIGHTRADAR24_API_KEY)
            await self._fr24_client.__aenter__()

    async def close_clients(self) -> None:
        """Close any opened API clients."""
        if self._aero_client:
            await self._aero_client.__aexit__(None, None, None)
            self._aero_client = None
        if self._fr24_client:
            await self._fr24_client.__aexit__(None, None, None)
            self._fr24_client = None

    async def validate_flight(self, flight: Dict) -> ValidationResult:
        """Enhanced validation that actively fills missing fields."""
        flight_no = flight.get("flight_no")
        flight_date = flight.get("date")

        if re.match(r"^A\d{5}R?$", flight_no or ""):
            # UPS internal pairing – external APIs won't match
            return self._apply_heuristics(flight)

        if not (flight_no and flight_date):
            return ValidationResult(
                is_valid=False,
                confidence=0.0,
                source="none",
                warnings=["Missing flight_no or date"],
            )

        # Track which fields are missing and need filling
        missing_fields = []
        if not flight.get("origin"):
            missing_fields.append("origin")
        if not flight.get("dest"):
            missing_fields.append("dest")
        if not flight.get("sched_out_local"):
            missing_fields.append("sched_out_local")
        if not flight.get("sched_in_local"):
            missing_fields.append("sched_in_local")

        logger.info(f"Validating {flight_no} on {flight_date}, missing fields: {missing_fields}")

        # Check if we should skip validation - all fields present
        if not missing_fields:
            logger.info(f"Skipping API validation for {flight_no} - all fields present")
            return ValidationResult(
                is_valid=True,
                confidence=1.0,
                source="prefilled",
                warnings=[],
                corrections={},
                filled_fields={}
            )

        api_results: Dict[str, Dict] = {}

        # Use shared clients when available to avoid reconnect overhead
        async def aero_task():
            if not self._has_aeroapi:
                return None
            if self._aero_client:
                return "aeroapi", await self._aero_client.search_flight(flight_no, flight_date)
            async with AeroAPIClient(FLIGHTAWARE_API_KEY) as client:
                return "aeroapi", await client.search_flight(flight_no, flight_date)

        async def fr24_task():
            if not self._has_fr24:
                return None
            if self._fr24_client:
                return "fr24", await self._fr24_client.search_flight(flight_no, flight_date)
            async with FR24Client(FLIGHTRADAR24_API_KEY) as client:
                return "fr24", await client.search_flight(flight_no, flight_date)

        # Gather all API results
        for res in await asyncio.gather(
            aero_task(), fr24_task(), return_exceptions=True
        ):
            if res and not isinstance(res, Exception) and res[1]:
                api_results[res[0]] = res[1]

        # Begin assembling ValidationResult
        result = ValidationResult(is_valid=False, confidence=0.0, source="none")

        # Process each API result
        if "aeroapi" in api_results:
            result = self._process_aeroapi_data(api_results["aeroapi"], flight, missing_fields)
            
        if "fr24" in api_results:
            fr24_res = self._process_fr24_data(api_results["fr24"], flight, missing_fields)
            # Merge results
            result.filled_fields.update(fr24_res.filled_fields)
            result.corrections.update(fr24_res.corrections)
            result.warnings.extend(fr24_res.warnings)
            result.confidence = max(result.confidence, fr24_res.confidence)
            result.source = (
                f"{result.source}+fr24" if result.source != "none" else "fr24"
            )
            if fr24_res.is_valid:
                result.is_valid = True

        # Firehose enrichment
        if FIREHOSE_USERNAME and FIREHOSE_PASSWORD and (missing_fields or not result.is_valid):
            try:
                logger.info("Enhancing %s with Firehose data…", flight_no)
                fh_data = await validate_with_firehose(flight)
                if fh_data:
                    fh_res = self._process_firehose_data(fh_data, flight, missing_fields)
                    result.filled_fields.update(fh_res.filled_fields)
                    result.corrections.update(fh_res.corrections)
                    result.confidence = max(result.confidence, fh_res.confidence)
                    result.source = (
                        f"{result.source}+firehose"
                        if result.source != "none"
                        else "firehose"
                    )
                    result.is_valid = True
            except Exception as exc:
                logger.error("Firehose enhancement error: %s", exc)

        # Cross-validate if we have data from multiple sources
        if len(api_results) > 1 or (api_results and "firehose" in result.source):
            result = self._cross_validate_sources(result, api_results, flight)

        # Fallback – heuristics
        if not result.is_valid and not result.filled_fields:
            result = self._apply_heuristics(flight)

        logger.info(
            "Validated %s via %s (%.2f) - Filled: %s", 
            flight_no, result.source, result.confidence,
            list(result.filled_fields.keys())
        )
        return result

    def _process_aeroapi_data(
        self, api_data: Dict, flight: Dict, missing_fields: List[str]
    ) -> ValidationResult:
        """Enhanced processing that fills missing fields from AeroAPI data."""
        vr = ValidationResult(is_valid=True, confidence=0.95, source="aeroapi")

        # Log the structure of api_data for debugging
        logger.debug(f"Processing AeroAPI data structure: {list(api_data.keys())}")

        # Extract times - check multiple possible field names
        sched_out = (
            api_data.get("scheduled_out") or 
            api_data.get("scheduled_off") or 
            api_data.get("scheduled_departure_time") or
            api_data.get("filed_departure_time") or
            api_data.get("departure_time")  # For schedules endpoint
        )

        origin = (
            _normalize_airport(api_data.get("origin")) or
            _normalize_airport(api_data.get("departure_airport")) or
            _to_iata(api_data.get("origin_iata")) or
            _to_iata(api_data.get("origin_icao"))
        )

        dest = (
            _normalize_airport(api_data.get("destination")) or
            _normalize_airport(api_data.get("arrival_airport")) or
            _to_iata(api_data.get("destination_iata")) or
            _to_iata(api_data.get("destination_icao"))
        )

        
        sched_in = (
            api_data.get("scheduled_in") or 
            api_data.get("scheduled_on") or 
            api_data.get("scheduled_arrival_time") or
            api_data.get("filed_arrival_time") or
            api_data.get("arrival_time")  # For schedules endpoint
        )
        
        # For schedules endpoint, times might be in a different format
        if not sched_out and "departure" in api_data:
            sched_out = api_data["departure"]
        if not sched_in and "arrival" in api_data:
            sched_in = api_data["arrival"]
        
        # Fill missing origin/dest
        if origin and "origin" in missing_fields:
            vr.filled_fields["origin"] = origin
        elif origin and flight.get("origin") and flight["origin"] != origin:
            vr.corrections["origin"] = origin
            
        if dest and "dest" in missing_fields:
            vr.filled_fields["dest"] = dest
        elif dest and flight.get("dest") and flight["dest"] != dest:
            vr.corrections["dest"] = dest
        
        # Fill missing times
        if sched_out and origin:
            time_str = convert_to_local_time(sched_out, origin)
            if time_str:
                if flight.get("sched_out_local") and flight["sched_out_local"] != time_str:
                    vr.corrections["sched_out_local"] = time_str
                elif "sched_out_local" in missing_fields:
                    vr.filled_fields["sched_out_local"] = time_str

        if sched_in and dest:
            time_str = convert_to_local_time(sched_in, dest)
            if time_str:
                if flight.get("sched_in_local") and flight["sched_in_local"] != time_str:
                    vr.corrections["sched_in_local"] = time_str
                elif "sched_in_local" in missing_fields:
                    vr.filled_fields["sched_in_local"] = time_str

        # Don't populate enriched_data anymore
        # Just log what we could have extracted
        logger.info(f"AeroAPI extraction - Origin: {origin}, Dest: {dest}, Times: {sched_out}/{sched_in}")
            
        return vr

    def _process_fr24_data(
        self, api_data: Dict, flight: Dict, missing_fields: List[str]
    ) -> ValidationResult:
        """Enhanced FR24 processing that fills missing fields."""
        vr = ValidationResult(is_valid=True, confidence=0.85, source="fr24")
        
        origin_raw = (
            api_data.get("airport", {}).get("origin", {}).get("code", {}).get("iata")
            or api_data.get("airport", {}).get("origin", {}).get("code", {}).get("icao")
        )
        origin = _to_iata(origin_raw)
        if origin:
           if flight.get("origin") and flight["origin"] != origin:
                vr.corrections["origin"] = origin
           elif "origin" in missing_fields:
                vr.filled_fields["origin"] = origin

        dest_raw = (
            api_data.get("airport", {}).get("destination", {}).get("code", {}).get("iata")
            or api_data.get("airport", {}).get("destination", {}).get("code", {}).get("icao")
        )
        dest = _to_iata(dest_raw)
        if dest:
            if flight.get("dest") and flight["dest"] != dest:
                vr.corrections["dest"] = dest
            elif "dest" in missing_fields:
                vr.filled_fields["dest"] = dest

        # Extract times
        time_data = api_data.get("time", {})
        sched_dep = time_data.get("scheduled", {}).get("departure")
        sched_arr = time_data.get("scheduled", {}).get("arrival")
        
        if sched_dep and "sched_out_local" in missing_fields:
            try:
                dt = datetime.fromtimestamp(sched_dep)
                vr.filled_fields["sched_out_local"] = dt.strftime("%H%M")
            except:
                pass
                
        if sched_arr and "sched_in_local" in missing_fields:
            try:
                dt = datetime.fromtimestamp(sched_arr)
                vr.filled_fields["sched_in_local"] = dt.strftime("%H%M")
            except:
                pass

        # Don't populate enriched_data
            
        return vr

    def _process_firehose_data(
        self, fh_data: Dict, flight: Dict, missing_fields: List[str]
    ) -> ValidationResult:
        """Enhanced Firehose processing that fills missing fields."""
        vr = ValidationResult(
            is_valid=True,
            confidence=fh_data.get("confidence", 1.0),
            source=fh_data.get("source", "firehose"),
        )
        
        # Map Firehose fields to our schema - only for missing core fields
        data = fh_data.get("data", {})
        
        # Fill origin if missing
        if "origin" in missing_fields and data.get("origin"):
            value = data["origin"][:3].upper()
            vr.filled_fields["origin"] = value
        elif data.get("origin") and flight.get("origin") != data["origin"][:3].upper():
            vr.corrections["origin"] = data["origin"][:3].upper()
            
        # Fill dest if missing
        if "dest" in missing_fields and data.get("destination"):
            value = data["destination"][:3].upper()
            vr.filled_fields["dest"] = value
        elif data.get("destination") and flight.get("dest") != data["destination"][:3].upper():
            vr.corrections["dest"] = data["destination"][:3].upper()
            
        # Fill scheduled times if missing
        if "sched_out_local" in missing_fields and data.get("sched_out_local"):
            vr.filled_fields["sched_out_local"] = data["sched_out_local"]
        if "sched_in_local" in missing_fields and data.get("sched_in_local"):
            vr.filled_fields["sched_in_local"] = data["sched_in_local"]
            
        # Don't populate enriched_data
            
        return vr

    def _cross_validate_sources(
        self, result: ValidationResult, api_results: Dict[str, Dict], flight: Dict
    ) -> ValidationResult:
        """Cross-validate data from multiple sources for consistency."""
        # Check for conflicts between sources
        conflicts = []
        
        # Compare filled fields and corrections
        all_fields = set(result.filled_fields.keys()) | set(result.corrections.keys())
        
        for field in all_fields:
            values = set()
            
            # Collect all proposed values for this field
            if field in result.filled_fields:
                values.add(result.filled_fields[field])
            if field in result.corrections:
                values.add(result.corrections[field])
                
            # If multiple different values, we have a conflict
            if len(values) > 1:
                conflicts.append(f"Conflicting {field} values: {values}")
                # Use the value with highest confidence source
                # Priority: firehose > aeroapi > fr24
                if "firehose" in result.source and field in result.filled_fields:
                    pass  # Keep firehose value
                elif "aeroapi" in result.source:
                    # Keep aeroapi value
                    pass
                else:
                    # Reduce confidence due to conflict
                    result.confidence *= 0.9
                    
        if conflicts:
            result.warnings.extend(conflicts)
            
        return result

    def _apply_heuristics(self, flight: Dict) -> ValidationResult:
        """Apply heuristics when API validation fails."""
        vr = ValidationResult(is_valid=True, confidence=0.50, source="heuristic")
        
        flight_no = flight.get("flight_no", "")
        
        # Basic validation
        if not flight_no:
            vr.is_valid = False
            vr.confidence = 0.0
            return vr
            
        # For UPS format
        if re.match(r"^A\d{5}[A-Z]?$", flight_no):
            vr.confidence = 0.85
            # Don't populate enriched_data
            
        # For standard airline format
        elif re.match(r"^[A-Z]{2}\d{1,4}[A-Z]?$", flight_no):
            vr.confidence = 0.60
            # Don't populate enriched_data
            
        return vr

    async def validate_schedule(self, flights: List[Dict]) -> Dict[str, Any]:
        """Validate a list of flight dicts concurrently."""
        start = time.time()

        await self.open_clients()
        try:
            tasks = [self.validate_flight(f) for f in flights]
            validations = await asyncio.gather(*tasks, return_exceptions=True)
        finally:
            await self.close_clients()

        enriched_flights: List[EnrichedFlight] = []
        warnings: List[str] = []

        for raw_flight, res in zip(flights, validations):
            if isinstance(res, Exception):
                logger.error("Validation task failed: %s", res)
                res = ValidationResult(
                    is_valid=False,
                    confidence=0.0,
                    source="error",
                    warnings=[str(res)],
                )

            # Create enriched flight with all original data
            ef = EnrichedFlight(**raw_flight, validation_result=res)
            
            # Apply filled fields FIRST (these were missing)
            for k, v in res.filled_fields.items():
                setattr(ef, k, v)
                
            # Then apply corrections (these override existing)
            for k, v in res.corrections.items():
                setattr(ef, k, v)
                
            # Don't add enriched_data to api_data anymore
                    
            enriched_flights.append(ef)
            warnings.extend(f"Flight {ef.flight_no}: {w}" for w in res.warnings)

        valid_count = sum(1 for e in enriched_flights if e.validation_result.is_valid)
        avg_conf = (
            sum(e.validation_result.confidence for e in enriched_flights)
            / len(enriched_flights)
            if enriched_flights
            else 0.0
        )

        # Summary of what was filled
        total_filled = sum(
            len(e.validation_result.filled_fields) for e in enriched_flights
        )

        return {
            "enriched_flights": [e.dict() for e in enriched_flights],
            "validation_summary": {
                "total_flights": len(flights),
                "valid_flights": valid_count,
                "average_confidence": avg_conf,
                "total_fields_filled": total_filled,
                "processing_time_seconds": time.time() - start,
                "sources_used": sorted(
                    {e.validation_result.source for e in enriched_flights}
                ),
                "warnings": warnings[:20],
            },
        }


# ─────────────────────────────────────────────────────────────────────────────
# ░ PUBLIC HELPERS ░
# ─────────────────────────────────────────────────────────────────────────────
async def validate_extraction_results(extraction_result: Dict) -> Dict:
    """Augment the extraction_result from GPT‑OCR pipeline with validation data."""
    validator = FlightValidator()
    flights_to_validate = extraction_result.get("flights", [])
    summary = await validator.validate_schedule(flights_to_validate)

    extraction_result["validation"] = summary["validation_summary"]
    extraction_result["enriched_flights"] = summary["enriched_flights"]

    if "quality_score" in extraction_result:
        validation_factor = summary["validation_summary"]["average_confidence"]
        extraction_result["quality_score"] = (
            0.7 * extraction_result["quality_score"] + 0.3 * validation_factor
        )

    return extraction_result


async def validate_flights_endpoint(flights: List[Dict]) -> Dict:
    """FastAPI helper – validate a list of flight dicts sent by client."""
    validator = FlightValidator()
    return await validator.validate_schedule(flights)