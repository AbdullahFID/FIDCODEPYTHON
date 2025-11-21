from __future__ import annotations

"""
flight_intel_patch package

Public API:
    - validate_extraction_results(extraction_result: Dict[str, Any]) -> Dict[str, Any]
    - FastFlightValidator
    - AeroAPIClient
    - FR24Client
    - ValidationResult
    - EnrichedFlight
"""

from .config import (
    FLIGHTAWARE_API_KEY,
    FLIGHTRADAR24_API_KEY,
    AVIATION_EDGE_API_KEY,
)
from .models import ValidationResult, EnrichedFlight
from .aeroapi_client import AeroAPIClient
from .fr24_client import FR24Client
from .validator import FastFlightValidator, validate_extraction_results

# Debug: show which API keys are configured (length only, never the values)
print("=" * 60)
print("🔑 API KEY DEBUG:")
print(
    f"   FLIGHTAWARE_API_KEY: {'SET' if FLIGHTAWARE_API_KEY else 'MISSING'} "
    f"(len={len(FLIGHTAWARE_API_KEY or '')})"
)
print(
    f"   FLIGHTRADAR24_API_KEY: {'SET' if FLIGHTRADAR24_API_KEY else 'MISSING'} "
    f"(len={len(FLIGHTRADAR24_API_KEY or '')})"
)
print(
    f"   AVIATION_EDGE_API_KEY: {'SET' if AVIATION_EDGE_API_KEY else 'MISSING'} "
    f"(len={len(AVIATION_EDGE_API_KEY or '')})"
)
print("=" * 60)

__all__ = [
    "validate_extraction_results",
    "FastFlightValidator",
    "AeroAPIClient",
    "FR24Client",
    "ValidationResult",
    "EnrichedFlight",
]
