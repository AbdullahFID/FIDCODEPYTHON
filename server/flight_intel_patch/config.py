from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

# Base URLs
AEROAPI_BASE_URL = "https://aeroapi.flightaware.com/aeroapi"  # v4 path
FR24_BASE_URL = "https://api.flightradar24.com/common/v1"
AVIATION_EDGE_BASE = "https://aviation-edge.com/v2/public"

# API keys
FLIGHTAWARE_API_KEY: str | None = os.getenv("FLIGHTAWARE_API_KEY")
FLIGHTRADAR24_API_KEY: str | None = os.getenv("FLIGHTRADAR24_API_KEY")
AVIATION_EDGE_API_KEY: str | None = os.getenv("AVIATION_EDGE_KEY")

# Rate limiting / concurrency knobs
AEROAPI_MAX_RPS: float = float(os.getenv("AEROAPI_MAX_RPS", "10"))
AEROAPI_BURST: int = int(os.getenv("AEROAPI_BURST", "3"))
VALIDATOR_CONCURRENCY: int = int(os.getenv("VALIDATOR_CONCURRENCY", "1"))

SERVICE_NAME = os.getenv("SERVICE_NAME", "flightintel")
ENV = os.getenv("APP_ENV", "dev")

# Minimal, high-hit map of US airports → tz
AIRPORT_TIMEZONES = {
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

# Cargo carriers that should prefer cargo-centric sources
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
