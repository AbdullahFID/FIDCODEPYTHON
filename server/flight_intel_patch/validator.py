from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

from dotenv import load_dotenv

from .config import (
    FLIGHTAWARE_API_KEY,
    FLIGHTRADAR24_API_KEY,
    AVIATION_EDGE_API_KEY,
)
from .models import ValidationResult, EnrichedFlight
from .aeroapi_client import AeroAPIClient
from .fr24_client import FR24Client

from logging_utils import log_event

load_dotenv()

logger = logging.getLogger("flightintel.validator")

# Simple in-memory cache:
# (flight_no_upper, date, has_origin, has_dest, has_time) -> ValidationResult
_VALIDATION_CACHE: Dict[
    Tuple[str, str, bool, bool, bool], Tuple[float, ValidationResult]
] = {}
_CACHE_TTL_S = 15 * 60
_CACHE_MAX = 1000


# ─────────────────────────────────────────────────────────────────────────────
# SMALL HELPERS (split ident, airport tz, etc.)
# ─────────────────────────────────────────────────────────────────────────────

import re as _re

_IDENT_SPLIT_RE = _re.compile(r"^([A-Z]{3}|[A-Z0-9]{2})?(\d{1,5})([A-Z]?)$")


def _split_ident(ident: str) -> Tuple[Optional[str], str, Optional[str]]:
    m = _IDENT_SPLIT_RE.match(ident.strip().upper())
    if not m:
        return None, ident.strip().upper(), None
    return (m.group(1), m.group(2), m.group(3) or None)


try:
    from zoneinfo import ZoneInfo
except ImportError:  # Py<3.9
    from backports.zoneinfo import ZoneInfo  # type: ignore

AIRPORT_TIMEZONES: Dict[str, str] = {
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


def _to_iata(code: Optional[str]) -> Optional[str]:
    if not code:
        return None
    c = code.upper().strip()
    if len(c) == 3:
        return c
    if len(c) == 4 and c.startswith("K"):
        return c[1:]
    return None


def _normalize_airport_fast(obj: Any) -> Optional[str]:
    if isinstance(obj, str):
        return _to_iata(obj)
    if isinstance(obj, dict):
        for key in ("code_iata", "iata", "code"):
            if key in obj:
                return _to_iata(str(obj[key]))
    return None


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
# CARGO CARRIER LIST (for routing to Aviation Edge later)
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# VALIDATOR
# ─────────────────────────────────────────────────────────────────────────────


class FastFlightValidator:
    """
    Validates & enriches each extracted flight.
    Uses FlightAware AeroAPI + FR24, and supports multiple matches (options list).
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

        log_event(
            logger,
            "validator_initialized",
            aeroapi_enabled=self._has_aero,
            fr24_enabled=self._has_fr24,
            av_edge_enabled=bool(AVIATION_EDGE_API_KEY),
        )

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
        is_cargo = airline in CARGO_CARRIERS if airline else False
        log_event(
            logger,
            "cargo_detection",
            flight_no=flight_no,
            airline=airline,
            is_cargo=is_cargo,
        )
        return is_cargo

    def _cache_get(self, flight_no: str, date: str, flight: Dict[str, Any]) -> Optional[ValidationResult]:
        has_origin = bool(flight.get("origin"))
        has_dest = bool(flight.get("dest"))
        has_time = bool(flight.get("sched_out_local"))
        key = (flight_no.upper(), date, has_origin, has_dest, has_time)
        entry = _VALIDATION_CACHE.get(key)
        if entry and (time.time() - entry[0] < _CACHE_TTL_S):
            log_event(
                logger,
                "validation_cache_hit",
                flight_no=flight_no,
                date=date,
                has_origin=has_origin,
                has_dest=has_dest,
                has_time=has_time,
            )
            return entry[1]
        return None

    def _cache_put(
        self,
        flight_no: str,
        date: str,
        flight: Dict[str, Any],
        result: ValidationResult,
    ) -> None:
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

    def _apply_aero_fields(
        self,
        api_data: Dict[str, Any],
        flight: Dict[str, Any],
        filled: Dict[str, Any],
        warnings: List[str],
    ) -> None:
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

        iso_dep = _first_iso(
            "scheduled_out", "scheduled_off", "departure_time", "departure"
        )
        iso_arr = _first_iso(
            "scheduled_in", "scheduled_on", "arrival_time", "arrival"
        )

        api_out_local = None
        api_in_local = None

        if iso_dep:
            api_out_local = _to_local_hhmm_from_iso(
                iso_dep, filled.get("origin") or flight.get("origin")
            )
        if iso_arr:
            api_in_local = _to_local_hhmm_from_iso(
                iso_arr, filled.get("dest") or flight.get("dest")
            )

        user_time = flight.get("sched_out_local")
        if user_time and api_out_local:
            try:
                user_minutes = int(user_time[:2]) * 60 + int(user_time[2:])
                api_minutes = int(api_out_local[:2]) * 60 + int(api_out_local[2:])
                time_diff = abs(api_minutes - user_minutes)

                if time_diff > 120:
                    log_event(
                        logger,
                        "validation_time_reject_aero",
                        level=logging.WARNING,
                        user_time=user_time,
                        api_time=api_out_local,
                        diff_minutes=time_diff,
                    )
                    filled.clear()
                    return
                elif time_diff > 15:
                    log_event(
                        logger,
                        "validation_time_uncertain_aero",
                        level=logging.WARNING,
                        user_time=user_time,
                        api_time=api_out_local,
                        diff_minutes=time_diff,
                    )
                    warnings.append(
                        f"Departure time differs by {time_diff} minutes from expected"
                    )
                else:
                    log_event(
                        logger,
                        "validation_time_ok_aero",
                        user_time=user_time,
                        api_time=api_out_local,
                        diff_minutes=time_diff,
                    )
            except Exception as e:
                log_event(
                    logger,
                    "validation_time_error_aero",
                    level=logging.WARNING,
                    error=str(e),
                )

        if api_out_local and not flight.get("sched_out_local"):
            filled["sched_out_local"] = api_out_local
        if api_in_local and not flight.get("sched_in_local"):
            filled["sched_in_local"] = api_in_local

    def _apply_fr24_fields(
        self,
        api_data: Dict[str, Any],
        flight: Dict[str, Any],
        filled: Dict[str, Any],
        warnings: List[str],
    ) -> None:
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

        user_time = flight.get("sched_out_local")
        if user_time and api_out_local:
            try:
                user_minutes = int(user_time[:2]) * 60 + int(user_time[2:])
                api_minutes = int(api_out_local[:2]) * 60 + int(api_out_local[2:])
                time_diff = abs(api_minutes - user_minutes)

                if time_diff > 120:
                    log_event(
                        logger,
                        "validation_time_reject_fr24",
                        level=logging.WARNING,
                        user_time=user_time,
                        api_time=api_out_local,
                        diff_minutes=time_diff,
                    )
                    filled.clear()
                    return
                elif time_diff > 15:
                    log_event(
                        logger,
                        "validation_time_uncertain_fr24",
                        level=logging.WARNING,
                        user_time=user_time,
                        api_time=api_out_local,
                        diff_minutes=time_diff,
                    )
                    warnings.append(
                        f"Departure time differs by {time_diff} minutes from expected"
                    )
                else:
                    log_event(
                        logger,
                        "validation_time_ok_fr24",
                        user_time=user_time,
                        api_time=api_out_local,
                        diff_minutes=time_diff,
                    )
            except Exception as e:
                log_event(
                    logger,
                    "validation_time_error_fr24",
                    level=logging.WARNING,
                    error=str(e),
                )

        if api_out_local and not flight.get("sched_out_local"):
            filled["sched_out_local"] = api_out_local
        if api_in_local and not flight.get("sched_in_local"):
            filled["sched_in_local"] = api_in_local

    async def _validate_one(self, flight: Dict[str, Any]) -> List[ValidationResult]:
        """
        Returns LIST of ValidationResults (multi-option support).
        """
        flight_no = (flight.get("flight_no") or "").strip().upper()
        date = (flight.get("date") or "").strip()

        log_event(
            logger,
            "flight_validation_started",
            flight_no=flight_no,
            date=date,
        )

        if not flight_no or not date:
            return [
                ValidationResult(
                    is_valid=False,
                    confidence=0.0,
                    source="none",
                    warnings=["missing_fields"],
                )
            ]

        cached = self._cache_get(flight_no, date, flight)
        if cached:
            return [cached]

        if all(
            flight.get(k)
            for k in ("origin", "dest", "sched_out_local", "sched_in_local")
        ):
            res = ValidationResult(is_valid=True, confidence=1.0, source="complete")
            self._cache_put(flight_no, date, flight, res)
            return [res]

        filled: Dict[str, Any] = {}
        source_tags: List[str] = []
        warnings: List[str] = []
        conf = 0.0

        is_cargo = self._is_cargo_flight(flight_no)

        if is_cargo:
            priority_order = ["fr24", "aeroapi"]
        else:
            priority_order = ["aeroapi", "fr24"]

        api_results: List[
            Tuple[str, Optional[Union[Dict[str, Any], List[Dict[str, Any]]]]]
        ] = []

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

        await asyncio.gather(_call_aero(), _call_fr24())

        api_results_sorted = sorted(
            api_results,
            key=lambda x: priority_order.index(x[0])
            if x[0] in priority_order
            else 999,
        )

        for tag, data in api_results_sorted:
            if not data:
                continue

            if isinstance(data, list) and len(data) > 1:
                log_event(
                    logger,
                    "flight_validation_multi_option",
                    provider=tag,
                    flight_no=flight_no,
                    date=date,
                    options=len(data),
                )

                all_results: List[ValidationResult] = []
                for idx, item in enumerate(data):
                    filled_temp: Dict[str, Any] = {}
                    warnings_temp: List[str] = []

                    if tag == "aeroapi":
                        self._apply_aero_fields(item, flight, filled_temp, warnings_temp)
                    elif tag == "fr24":
                        self._apply_fr24_fields(item, flight, filled_temp, warnings_temp)

                    if filled_temp:
                        conf_temp = 0.95 if tag == "aeroapi" else 0.85
                        warnings_temp.append(
                            f"Option {idx + 1} of {len(data)}"
                        )
                        all_results.append(
                            ValidationResult(
                                is_valid=True,
                                confidence=conf_temp * 0.8,
                                source=tag,
                                filled_fields=filled_temp,
                                warnings=warnings_temp,
                            )
                        )

                if all_results:
                    return all_results
                continue

            if isinstance(data, list):
                data = data[0] if data else None
                if not data:
                    continue

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

            has_all = all(
                filled.get(k) or flight.get(k)
                for k in ("origin", "dest", "sched_out_local", "sched_in_local")
            )
            if has_all:
                log_event(
                    logger,
                    "flight_validation_complete",
                    flight_no=flight_no,
                    date=date,
                    provider_sequence="+".join(source_tags),
                )
                break

        if not source_tags:
            log_event(
                logger,
                "flight_validation_no_api_data",
                flight_no=flight_no,
                date=date,
            )
            res = ValidationResult(
                is_valid=True,
                confidence=0.5,
                source="heuristic",
                filled_fields={},
                warnings=["No API data found"],
            )
            self._cache_put(flight_no, date, flight, res)
            return [res]

        res = ValidationResult(
            is_valid=True,
            confidence=conf if conf > 0 else 0.5,
            source="+".join(source_tags),
            filled_fields=filled,
            warnings=warnings,
        )
        self._cache_put(flight_no, date, flight, res)
        return [res]

    async def validate_batch(self, flights: List[Dict[str, Any]]) -> Dict[str, Any]:
        start = time.time()
        need: List[Dict[str, Any]] = []
        prefilled: List[List[ValidationResult]] = []

        for f in flights:
            if all(
                f.get(k)
                for k in ("origin", "dest", "sched_out_local", "sched_in_local")
            ):
                prefilled.append(
                    [ValidationResult(is_valid=True, confidence=1.0, source="complete")]
                )
            else:
                need.append(f)

        log_event(
            logger,
            "validation_batch_started",
            total_flights=len(flights),
            needing_validation=len(need),
        )

        validations: List[List[ValidationResult]] = []

        if need:
            for i, f in enumerate(need):
                try:
                    if i > 0 and i % 3 == 0:
                        await asyncio.sleep(1.0)
                    results = await self._validate_one(f)
                    validations.append(results)
                except Exception as e:
                    log_event(
                        logger,
                        "validation_exception",
                        level=logging.ERROR,
                        error=str(e),
                    )
                    validations.append(
                        [
                            ValidationResult(
                                is_valid=False,
                                confidence=0.0,
                                source="error",
                                warnings=[str(e)],
                            )
                        ]
                    )

        enriched: List[EnrichedFlight] = []

        # For simplicity, pair flights with validations in order
        for raw, val_list in zip(
            flights, validations if validations else prefilled
        ):
            for val in val_list:
                ef = EnrichedFlight(**raw, validation_result=val)
                for k, v in (val.filled_fields or {}).items():
                    setattr(ef, k, v)
                enriched.append(ef)

        valid_count = sum(
            1
            for e in enriched
            if e.validation_result and e.validation_result.is_valid
        )
        avg_conf = (
            sum(
                e.validation_result.confidence
                for e in enriched
                if e.validation_result
            )
            / len(enriched)
            if enriched
            else 0.0
        )
        total_filled = sum(
            len(e.validation_result.filled_fields)
            for e in enriched
            if e.validation_result
        )
        elapsed = time.time() - start

        log_event(
            logger,
            "validation_batch_completed",
            flights_input=len(flights),
            flights_output=len(enriched),
            valid_flights=valid_count,
            average_confidence=avg_conf,
            total_fields_filled=total_filled,
            duration_ms=int(elapsed * 1000),
        )

        return {
            "enriched_flights": [e.dict() for e in enriched],
            "validation_summary": {
                "total_input_flights": len(flights),
                "total_output_flights": len(enriched),
                "valid_flights": valid_count,
                "average_confidence": avg_conf,
                "total_fields_filled": total_filled,
                "processing_time_seconds": elapsed,
                "sources_used": sorted(
                    set(
                        (
                            e.validation_result.source
                            if e.validation_result
                            else "none"
                        )
                        for e in enriched
                    )
                ),
            },
        }


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC ENTRYPOINT
# ─────────────────────────────────────────────────────────────────────────────


async def validate_extraction_results(
    extraction_result: Dict[str, Any]
) -> Dict[str, Any]:
    log_event(logger, "validation_request_received")

    flights = extraction_result.get("flights", [])

    async with FastFlightValidator() as validator:
        summary = await validator.validate_batch(flights)

    extraction_result["validation"] = summary["validation_summary"]
    extraction_result["enriched_flights"] = summary["enriched_flights"]

    log_event(logger, "validation_request_completed")
    return extraction_result
