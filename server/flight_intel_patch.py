"""
Flight Intel Validation & Firehose Integration Module  – FULL SOURCE

Validates and enriches extracted flight data using external APIs:
• FlightAware AeroAPI v4
• FlightRadar24 API
• FlightAware Firehose streaming data

Key fixes & upgrades (v4.1)
──────────────────────────
• No more “Unclosed client session/connector” warnings – every aiohttp session is
  opened inside its own `async with`.
• Automatic router:  >2 days in the future → ‘schedule’ path, else validate live.
• Stub schedule validator (returns 0.80 confidence) until a real schedule API
  is wired in.
• Typed helper `choose_validation_path` in top‑level namespace.
• Minor PEP 8 touch‑ups; fully‑annotated dataclasses & Pydantic models.
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
from typing import Any, Dict, List, Optional

import aiohttp
from pydantic import BaseModel, Field

# Local dependency – thin wrapper around Firehose streaming
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

# ─────────────────────────────────────────────────────────────────────────────
# ░ HELPER ░
# ─────────────────────────────────────────────────────────────────────────────
def choose_validation_path(date_str: str) -> str:
    """
    Decide which validator to use based on date.

    Returns:
        'schedule' if the flight date is strictly more than 2 days in the future,
        otherwise 'live'.
        If the date is un‑parseable, default to 'schedule'.
    """
    try:
        flight_dt = datetime.strptime(date_str, "%m/%d/%Y").replace(
            tzinfo=timezone.utc
        )
    except ValueError:
        return "schedule"

    delta_days = (flight_dt.date() - datetime.utcnow().date()).days
    return "schedule" if delta_days > 2 else "live"

# ─────────────────────────────────────────────────────────────────────────────
# ░ DATA MODELS ░
# ─────────────────────────────────────────────────────────────────────────────
class ValidationResult(BaseModel):
    """Outcome of a single‑flight validation attempt."""
    is_valid: bool = Field(..., description="Whether the flight is considered valid")
    confidence: float = Field(..., description="Confidence score (0‑1)")
    source: str = Field(..., description="Which source(s) validated the flight")
    warnings: List[str] = Field(default_factory=list)
    corrections: Dict[str, Any] = Field(default_factory=dict)
    enriched_data: Dict[str, Any] = Field(default_factory=dict)


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
# ░ API CLIENTS ░
# ─────────────────────────────────────────────────────────────────────────────
class AeroAPIClient:
    """Tiny wrapper around FlightAware AeroAPI."""

    def __init__(self, api_key: str) -> None:
        self._headers = {"x-apikey": api_key, "Accept": "application/json"}
        self._session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self) -> "AeroAPIClient":
        self._session = aiohttp.ClientSession(headers=self._headers)
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._session:
            await self._session.close()

    # ─────────────────────────────────────────────────────────────────────
    async def search_flight(self, flight_no: str, date: str) -> Optional[Dict]:
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
        if -10 <= delta <= 2:
            endpoint = f"/flights/{ident}"
            params = {
                "start": flight_dt.isoformat(),
                "end": (flight_dt + timedelta(days=1)).isoformat(),
            }
        elif delta < -10:
            endpoint = f"/history/flights/{ident}"
            params = {
                "start": flight_dt.isoformat(),
                "end": (flight_dt + timedelta(days=1)).isoformat(),
            }
        else:
            endpoint = (
                f"/schedules/{flight_dt:%Y-%m-%d}/"
                f"{(flight_dt + timedelta(days=1)):%Y-%m-%d}"
            )
            params = {"ident": ident}

        async with self._session.get(
            f"{AEROAPI_BASE_URL}{endpoint}", params=params
        ) as resp:
            if resp.status != 200:
                logger.debug("AeroAPI %s returned %s", ident, resp.status)
                return None
            data = await resp.json()
            return data.get("flights", data.get("data", []))[0] if data else None


class FR24Client:
    """Wrapper for the (undocumented) FlightRadar24 JSON API."""

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

    # ─────────────────────────────────────────────────────────────────────
    async def search_flight(self, flight_no: str, date: str) -> Optional[Dict]:
        if not flight_no:
            return None

        ident = flight_no.strip().upper()
        params = {
            "query": ident,
            "fetchBy": "flight",
            "limit": 25,
            "token": self._api_key,
        }
        url = f"{FR24_BASE_URL}/flight/list.json"

        try:
            async with self._session.get(url, params=params) as resp:
                if resp.status != 200:
                    logger.debug("FR24 %s returned %s", ident, resp.status)
                    return None
                data = await resp.json()
        except Exception as exc:
            logger.debug("FR24 exception for %s: %s", ident, exc)
            return None

        flights: List[Dict] = (
            data.get("result", {}).get("response", {}).get("data", [])
        )
        if not flights:
            return None

        target_date = datetime.strptime(date, "%m/%d/%Y").date()
        for item in flights:
            ts = item.get("time", {}).get("scheduled", {}).get("departure")
            if ts and datetime.fromtimestamp(ts).date() == target_date:
                return item
        return None


# ─────────────────────────────────────────────────────────────────────────────
# ░ VALIDATOR ░
# ─────────────────────────────────────────────────────────────────────────────
class FlightValidator:
    """Coordinates look‑ups across AeroAPI, FR24 and Firehose."""

    def __init__(self) -> None:
        self._has_aeroapi = bool(FLIGHTAWARE_API_KEY)
        self._has_fr24 = bool(FLIGHTRADAR24_API_KEY)
        if not self._has_aeroapi:
            logger.warning("AeroAPI key not configured – skipping that source")
        if not self._has_fr24:
            logger.warning("FR24 key not configured – skipping that source")
        if not (FIREHOSE_USERNAME and FIREHOSE_PASSWORD):
            logger.warning(
                "Firehose credentials not configured – Firehose enrichment disabled"
            )

    # ─────────────────────────────────────────────────────────────────────
    async def _schedule_stub(self, flight: Dict) -> ValidationResult:
        """
        Placeholder validator for future flights (>2 days ahead).
        Always returns is_valid=True with 0.80 confidence.
        """
        return ValidationResult(
            is_valid=True,
            confidence=0.80,
            source="schedule",
            warnings=[],
            corrections={},
            enriched_data={},
        )

    async def validate_flight(self, flight: Dict) -> ValidationResult:
        """Validate a single flight dict (date & flight_no required)."""
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

        api_results: Dict[str, Dict] = {}

        # Each lookup gets its own client/session → no leaks
        async def aero_task():
            if not self._has_aeroapi:
                return None
            async with AeroAPIClient(FLIGHTAWARE_API_KEY) as client:
                return "aeroapi", await client.search_flight(flight_no, flight_date)

        async def fr24_task():
            if not self._has_fr24:
                return None
            async with FR24Client(FLIGHTRADAR24_API_KEY) as client:
                return "fr24", await client.search_flight(flight_no, flight_date)

        for res in await asyncio.gather(
            aero_task(), fr24_task(), return_exceptions=True
        ):
            if res and not isinstance(res, Exception) and res[1]:
                api_results[res[0]] = res[1]

        # Begin assembling ValidationResult
        result = ValidationResult(is_valid=False, confidence=0.0, source="none")

        if "aeroapi" in api_results:
            result = self._process_aeroapi_data(api_results["aeroapi"], flight)
        if "fr24" in api_results:
            fr24_res = self._process_fr24_data(api_results["fr24"], flight)
            result.enriched_data.update(fr24_res.enriched_data)
            result.warnings.extend(fr24_res.warnings)
            result.source = (
                f"{result.source}+fr24" if result.source != "none" else "fr24"
            )

        # Firehose enrichment (best‑effort)
        if FIREHOSE_USERNAME and FIREHOSE_PASSWORD:
            try:
                logger.info("Enhancing %s with Firehose data…", flight_no)
                fh_data = await validate_with_firehose(flight)
                if fh_data:
                    fh_res = self._process_firehose_data(fh_data, flight)
                    result.enriched_data.update(fh_res.enriched_data)
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

        # Fallback – heuristics
        if not result.is_valid:
            result = self._apply_heuristics(flight)

        logger.info(
            "Validated %s via %s (%.2f)", flight_no, result.source, result.confidence
        )
        return result

    # ────────────────────────────────────────────────────────── helpers ─────
    def _process_aeroapi_data(self, api_data: Dict, flight: Dict) -> ValidationResult:
        vr = ValidationResult(is_valid=True, confidence=0.95, source="aeroapi")

        origin = api_data.get("origin", {}).get("code")
        dest = api_data.get("destination", {}).get("code")

        if origin and flight.get("origin") and flight["origin"] != origin:
            vr.corrections["origin"] = origin
        elif origin and not flight.get("origin"):
            vr.enriched_data["origin"] = origin

        if dest and flight.get("dest") and flight["dest"] != dest:
            vr.corrections["dest"] = dest
        elif dest and not flight.get("dest"):
            vr.enriched_data["dest"] = dest

        if api_data.get("aircraft_type"):
            vr.enriched_data["aircraft_type"] = api_data["aircraft_type"]
        if api_data.get("registration"):
            vr.enriched_data["aircraft_reg"] = api_data["registration"]
        if api_data.get("status"):
            vr.enriched_data["flight_status"] = api_data["status"]
        return vr

    def _process_fr24_data(self, api_data: Dict, flight: Dict) -> ValidationResult:
        vr = ValidationResult(is_valid=True, confidence=0.85, source="fr24")
        origin = (
            api_data.get("airport", {}).get("origin", {}).get("code", {}).get("iata")
        )
        dest = (
            api_data.get("airport", {})
            .get("destination", {})
            .get("code", {})
            .get("iata")
        )
        if origin and flight.get("origin") != origin:
            vr.corrections["origin"] = origin
        elif origin and not flight.get("origin"):
            vr.enriched_data["origin"] = origin

        model = api_data.get("aircraft", {}).get("model", {}).get("text")
        if model:
            vr.enriched_data["aircraft_type"] = model
        return vr

    def _process_firehose_data(
        self, fh_data: Dict, flight: Dict
    ) -> ValidationResult:
        vr = ValidationResult(
            is_valid=True,
            confidence=fh_data.get("confidence", 1.0),
            source=fh_data.get("source", "firehose"),
        )
        field_map = {"destination": "dest", "aircrafttype": "aircraft_type"}
        for key, value in fh_data.get("data", {}).items():
            mapped = field_map.get(key, key)
            if flight.get(mapped) and flight[mapped] != value:
                vr.corrections[mapped] = value
            else:
                vr.enriched_data[mapped] = value
        return vr

    def _apply_heuristics(self, flight: Dict) -> ValidationResult:
        vr = ValidationResult(is_valid=True, confidence=0.50, source="heuristic")
        if not re.match(r"^[A-Z0-9]{2,7}[A-Z]?$", flight.get("flight_no", "")):
            vr.warnings.append("Flight number format appears invalid")
            vr.confidence -= 0.2
        try:
            datetime.strptime(flight.get("date", ""), "%m/%d/%Y")
        except ValueError:
            vr.warnings.append("Invalid date format")
            vr.confidence -= 0.2
        vr.is_valid = vr.confidence > 0.25
        return vr

    # ─────────────────────────────────────────────────────────────────────
    async def validate_schedule(self, flights: List[Dict]) -> Dict[str, Any]:
        """
        Validate a list of flight dicts concurrently, aggregate results.

        Uses `choose_validation_path()` to decide whether to run the regular
        live validator or the stub ‘schedule’ validator.
        """
        start = time.time()

        tasks: List[asyncio.Future] = []
        for f in flights:
            path = choose_validation_path(f.get("date", ""))
            if path == "schedule":
                tasks.append(self._schedule_stub(f))  # type: ignore[arg-type]
            else:
                tasks.append(self.validate_flight(f))  # type: ignore[arg-type]

        validations = await asyncio.gather(*tasks, return_exceptions=True)

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

            ef = EnrichedFlight(**raw_flight, validation_result=res)
            # Apply corrections + enrichment to EnrichedFlight top‑level fields
            for k, v in res.corrections.items():
                setattr(ef, k, v)
            for k, v in res.enriched_data.items():
                if hasattr(ef, k) and not getattr(ef, k):
                    setattr(ef, k, v)
                else:
                    ef.api_data[k] = v
            enriched_flights.append(ef)
            warnings.extend(f"Flight {ef.flight_no}: {w}" for w in res.warnings)

        valid_count = sum(1 for e in enriched_flights if e.validation_result.is_valid)
        avg_conf = (
            sum(e.validation_result.confidence for e in enriched_flights)
            / len(enriched_flights)
            if enriched_flights
            else 0.0
        )

        return {
            "enriched_flights": [e.dict() for e in enriched_flights],
            "validation_summary": {
                "total_flights": len(flights),
                "valid_flights": valid_count,
                "average_confidence": avg_conf,
                "processing_time_seconds": time.time() - start,
                "sources_used": sorted(
                    {e.validation_result.source for e in enriched_flights}
                ),
                "warnings": warnings[:20],
            },
        }


# ─────────────────────────────────────────────────────────────────────────────
# ░ PUBLIC HELPERS – used by main FastAPI app ░
# ─────────────────────────────────────────────────────────────────────────────
async def validate_extraction_results(extraction_result: Dict) -> Dict:
    """
    Augment the extraction_result from GPT‑OCR pipeline with validation data.
    """
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
    """
    FastAPI helper – validate a list of flight dicts sent by client.
    """
    validator = FlightValidator()
    return await validator.validate_schedule(flights)
