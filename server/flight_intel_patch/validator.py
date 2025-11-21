from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union

from .config import (
    CARGO_CARRIERS,
    FLIGHTAWARE_API_KEY,
    FLIGHTRADAR24_API_KEY,
)
from .aeroapi_client import AeroAPIClient
from .fr24_client import FR24Client
from .models import EnrichedFlight, ValidationResult
from .utils import (
    _normalize_airport_fast,
    _split_ident,
    _to_local_hhmm_from_epoch_utc,
    _to_local_hhmm_from_iso,
)

logger = logging.getLogger("flight-validator")

# Cache: (flight_no_upper, date, has_origin, has_dest, has_time) -> (ts, ValidationResult)
_VALIDATION_CACHE: Dict[
    Tuple[str, str, bool, bool, bool], Tuple[float, ValidationResult]
] = {}
_CACHE_TTL_S = 15 * 60
_CACHE_MAX = 1000


class FastFlightValidator:
    """
    Validates & enriches each extracted flight.
    """

    def __init__(self) -> None:
        self._has_aero = bool(FLIGHTAWARE_API_KEY)
        self._has_fr24 = bool(FLIGHTRADAR24_API_KEY)

        self._aero: Optional[AeroAPIClient] = (
            AeroAPIClient(FLIGHTAWARE_API_KEY)
            if self._has_aero and FLIGHTAWARE_API_KEY
            else None
        )
        self._fr24: Optional[FR24Client] = (
            FR24Client(FLIGHTRADAR24_API_KEY)
            if self._has_fr24 and FLIGHTRADAR24_API_KEY
            else None
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
        is_cargo = airline in CARGO_CARRIERS if airline else False
        logger.info(
            f"   🔍 Cargo detection: '{flight_no}' -> airline='{airline}' -> is_cargo={is_cargo}"
        )
        if airline and not is_cargo:
            logger.info(
                f"      Sample cargo carriers: {sorted(CARGO_CARRIERS)[:10]}"
            )
        return is_cargo

    def _cache_get(
        self, flight_no: str, date: str, flight: Dict[str, Any]
    ) -> Optional[ValidationResult]:
        """Cache key includes data completeness to avoid wrong hits."""
        has_origin = bool(flight.get("origin"))
        has_dest = bool(flight.get("dest"))
        has_time = bool(flight.get("sched_out_local"))

        key = (flight_no.upper(), date, has_origin, has_dest, has_time)
        entry = _VALIDATION_CACHE.get(key)
        if entry and (time.time() - entry[0] < _CACHE_TTL_S):
            logger.info(f"   📦 Cache hit for {flight_no} on {date}")
            return entry[1]
        return None

    def _cache_put(
        self, flight_no: str, date: str, flight: Dict[str, Any], result: ValidationResult
    ) -> None:
        """Cache key includes data completeness."""
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
        Returns LIST of ValidationResults.
        - Single item list for normal flights
        - Multiple items when multiple matches found
        """
        flight_no = (flight.get("flight_no") or "").strip().upper()
        date = (flight.get("date") or "").strip()

        logger.info(f"🔍 Validating {flight_no} on {date}")

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
            logger.info("   🚛 Cargo flight - priority: FR24 → FlightAware")
            priority_order = ["fr24", "aeroapi"]
        else:
            logger.info("   ✈️ Passenger flight - priority: FlightAware → FR24")
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
            data = await self._fr24.search(
                flight_no,
                date,
                origin_hint=_normalize_airport_fast(flight.get("origin")),
                dest_hint=_normalize_airport_fast(flight.get("dest")),
            )
            api_results.append(("fr24", data))

        await asyncio.gather(_call_aero(), _call_fr24())

        api_results_sorted = sorted(
            api_results,
            key=lambda x: priority_order.index(x[0]) if x[0] in priority_order else 999,
        )

        # Handle multiple result sets
        for tag, data in api_results_sorted:
            if not data:
                continue

            if isinstance(data, list) and len(data) > 1:
                logger.info(f"   📋 {tag} returned {len(data)} options - returning all")
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
                                source=f"{tag}",
                                filled_fields=filled_temp,
                                warnings=warnings_temp,
                            )
                        )

                if all_results:
                    logger.info(
                        f"   ✅ Returning {len(all_results)} options to caller"
                    )
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
                logger.info(f"   ✅ All fields complete after {tag}")
                break

        if not source_tags:
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

    def _apply_aero_fields(
        self,
        api_data: Dict[str, Any],
        flight: Dict[str, Any],
        filled: Dict[str, Any],
        warnings: List[str],
    ) -> None:
        """
        Extract origin/dest + scheduled times from AeroAPI record.
        Works with both /flights and /schedules shapes.
        """
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
                iso_dep, origin or flight.get("origin")
            )
        if iso_arr:
            api_in_local = _to_local_hhmm_from_iso(
                iso_arr, dest or flight.get("dest")
            )

        user_time = flight.get("sched_out_local")
        if user_time and api_out_local:
            try:
                user_minutes = int(user_time[:2]) * 60 + int(user_time[2:])
                api_minutes = int(api_out_local[:2]) * 60 + int(api_out_local[2:])
                time_diff = abs(api_minutes - user_minutes)

                if time_diff > 120:
                    logger.warning(
                        f"   ⚠️  Time mismatch: user={user_time} api={api_out_local} "
                        f"(diff={time_diff}min) - rejecting (>2hrs)"
                    )
                    filled.clear()
                    return
                elif time_diff > 15:
                    logger.warning(
                        f"   ⚠️  Time mismatch: user={user_time} api={api_out_local} "
                        f"(diff={time_diff}min) - uncertain"
                    )
                    warnings.append(
                        f"Departure time differs by {time_diff} minutes from expected"
                    )
                else:
                    logger.info(
                        f"   ✅ Time validated: user={user_time} api={api_out_local} "
                        f"(diff={time_diff}min)"
                    )
            except Exception as e:
                logger.warning(f"   ⚠️  Time validation error: {e}")

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
        """
        Extract fields from FR24 record.
        """
        airport = api_data.get("airport", {}) if isinstance(api_data, dict) else {}
        o_code = airport.get("origin", {}).get("code", {})
        d_code = airport.get("destination", {}).get("code", {})

        origin = _normalize_airport_fast(
            o_code.get("iata") or o_code.get("icao") or o_code
        )
        dest = _normalize_airport_fast(
            d_code.get("iata") or d_code.get("icao") or d_code
        )

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
                dep_epoch, origin or flight.get("origin")
            )
        if arr_epoch:
            api_in_local = _to_local_hhmm_from_epoch_utc(
                arr_epoch, dest or flight.get("dest")
            )

        user_time = flight.get("sched_out_local")
        if user_time and api_out_local:
            try:
                user_minutes = int(user_time[:2]) * 60 + int(user_time[2:])
                api_minutes = int(api_out_local[:2]) * 60 + int(api_out_local[2:])
                time_diff = abs(api_minutes - user_minutes)

                if time_diff > 120:
                    logger.warning(
                        f"   ⚠️  FR24 time mismatch: user={user_time} api={api_out_local} "
                        f"(diff={time_diff}min) - rejecting (>2hrs)"
                    )
                    filled.clear()
                    return
                elif time_diff > 15:
                    logger.warning(
                        f"   ⚠️  FR24 time mismatch: user={user_time} api={api_out_local} "
                        f"(diff={time_diff}min) - uncertain"
                    )
                    warnings.append(
                        f"Departure time differs by {time_diff} minutes from expected"
                    )
                else:
                    logger.info(
                        f"   ✅ FR24 time validated: user={user_time} api={api_out_local} "
                        f"(diff={time_diff}min)"
                    )
            except Exception as e:
                logger.warning(f"   ⚠️  FR24 time validation error: {e}")

        if api_out_local and not flight.get("sched_out_local"):
            filled["sched_out_local"] = api_out_local

        if api_in_local and not flight.get("sched_in_local"):
            filled["sched_in_local"] = api_in_local

    async def validate_batch(self, flights: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate a batch of flights and return enriched result + summary.
        """
        start = time.time()

        validations_per_flight: List[List[ValidationResult]] = []

        logger.info("🚀 BATCH VALIDATION STARTED")
        logger.info(f"   Total flights: {len(flights)}")

        for idx, f in enumerate(flights):
            if all(
                f.get(k)
                for k in ("origin", "dest", "sched_out_local", "sched_in_local")
            ):
                validations_per_flight.append(
                    [
                        ValidationResult(
                            is_valid=True, confidence=1.0, source="complete"
                        )
                    ]
                )
                continue

            if idx > 0 and idx % 3 == 0:
                await asyncio.sleep(1.0)

            try:
                results = await self._validate_one(f)
                validations_per_flight.append(results)
            except Exception as e:
                logger.error(f"Validation exception: {e}")
                validations_per_flight.append(
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

        for raw, val_list in zip(flights, validations_per_flight):
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

        logger.info("✅ BATCH VALIDATION COMPLETE")
        logger.info(
            f"   Enriched flights: {len(enriched)} from {len(flights)} input | "
            f"{elapsed:.2f}s"
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


async def validate_extraction_results(
    extraction_result: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Public entrypoint: accepts extractor output and returns the same dict updated with:
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
