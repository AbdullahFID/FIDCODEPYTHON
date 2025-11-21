from __future__ import annotations

import io
import logging
import time
import base64
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Query, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel
from starlette.status import HTTP_400_BAD_REQUEST

from openai import AsyncOpenAI

from flight_intel_patch.validator import validate_extraction_results
from logging_utils import configure_logging, new_request_id, log_event

# ------------------------------------------------------------------------------
# APP + LOGGING SETUP
# ------------------------------------------------------------------------------

configure_logging()
logger = logging.getLogger("flightintel.api")

app = FastAPI(title="Flight-Intel v8.1", version="8.1.0")

# Basic CORS (adjust origins for your environment)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OpenAI client
client = AsyncOpenAI()

MODEL_NAME = "gpt-5.1"
OPENAI_TIMEOUT_SECONDS = 30
WORKER_COUNT = 16

logger.info(
    "Config: model=%s, timeout=%ss, workers=%s",
    MODEL_NAME,
    OPENAI_TIMEOUT_SECONDS,
    WORKER_COUNT,
)
logger.info("Starting Flight-Intel v8.1 server")


# ------------------------------------------------------------------------------
# SHARED MODELS
# ------------------------------------------------------------------------------

class Flight(BaseModel):
    date: str
    flight_no: str
    origin: Optional[str] = None
    dest: Optional[str] = None
    sched_out_local: Optional[str] = None
    sched_in_local: Optional[str] = None
    page_number: Optional[int] = None
    confidence: float = 1.0


class ExtractionResponse(BaseModel):
    flights: List[Flight]
    connections: List[Dict[str, Any]] = []
    total_flights_found: int
    avg_conf: float  # 🔥 Changed from avg_confidence for frontend compatibility
    processing_time: Dict[str, float]
    extraction_method: str
    metadata: Dict[str, Any]
    validation: Optional[Dict[str, Any]] = None
    enriched_flights: Optional[List[Dict[str, Any]]] = None
    cost_analysis: Optional[Dict[str, Any]] = None


# ------------------------------------------------------------------------------
# REQUEST LOGGING MIDDLEWARE (Loki-ready)
# ------------------------------------------------------------------------------

@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    rid = new_request_id()
    start = time.time()

    log_event(
        logger,
        "http_request_started",
        method=request.method,
        path=request.url.path,
        client_ip=request.client.host if request.client else None,
        user_agent=request.headers.get("user-agent"),
        request_id=rid,
    )

    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    finally:
        duration_ms = int((time.time() - start) * 1000)
        log_event(
            logger,
            "http_request_finished",
            method=request.method,
            path=request.url.path,
            status_code=status_code,
            duration_ms=duration_ms,
            request_id=rid,
        )


# ------------------------------------------------------------------------------
# IMAGE QUALITY ANALYSIS
# ------------------------------------------------------------------------------

def _analyse_image_quality(image_bytes: bytes) -> Dict[str, Any]:
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("L")
        w, h = img.size
        pixels = img.load()

        # Simple sharpness metric
        total_diff = 0
        count = 0
        for y in range(1, h):
            for x in range(1, w):
                total_diff += abs(pixels[x, y] - pixels[x - 1, y])
                total_diff += abs(pixels[x, y] - pixels[x, y - 1])
                count += 2
        sharpness = total_diff / max(count, 1)

        # Simple contrast proxy
        mean = sum(pixels[x, y] for x in range(w) for y in range(h)) / float(w * h)
        var = sum((pixels[x, y] - mean) ** 2 for x in range(w) for y in range(h)) / float(
            w * h
        )
        contrast = var ** 0.5

        return {
            "sharp": round(sharpness, 1),
            "contrast": round(contrast, 1),
            "grid": w > 800 and h > 800,
            "text_regions": int((w * h) / 2000),
        }
    except Exception as e:
        log_event(logger, "image_quality_error", level=logging.WARNING, error=str(e))
        return {
            "sharp": None,
            "contrast": None,
            "grid": None,
            "text_regions": None,
        }


# ------------------------------------------------------------------------------
# OPENAI VISION (CORRECTED + STRONG CONSTRAINTS)
# ------------------------------------------------------------------------------

async def _openai_extract_flights(
    image_bytes: bytes,
    airline: Optional[str],
) -> Dict[str, Any]:

    import json

    t0 = time.time()

    # Base64 encode the image
    b64 = base64.b64encode(image_bytes).decode("ascii")
    image_url = f"data:image/jpeg;base64,{b64}"

    system_prompt = (
        "You are a flight schedule extraction engine.\n"
        "You receive an image (or PDF page) of a bid award / roster / schedule.\n\n"
        "Return ONLY valid JSON with this structure:\n"
        "{\n"
        '  \"flights\": [\n'
        "    {\n"
        '      \"date\": \"MM/DD/YYYY\",  // REQUIRED\n'
        '      \"flight_no\": \"AA1234\", // REQUIRED, non-empty\n'
        '      \"origin\": \"JFK\" or null,\n'
        '      \"dest\": \"LAX\" or null,\n'
        '      \"sched_out_local\": \"HHMM\" or null,\n'
        '      \"sched_in_local\": \"HHMM\" or null,\n'
        '      \"page_number\": 1,\n'
        '      \"confidence\": 0.0\n'
        "    }\n"
        "  ],\n"
        '  \"connections\": []\n'
        "}\n\n"
        "CRITICAL RULES:\n"
        "1) If the flight number cannot be read, DO NOT include that row in `flights`.\n"
        "2) `flight_no` must NEVER be empty or missing for any flight.\n"
        "3) If any other field is unknown, set it to null.\n"
    )

    if airline:
        system_prompt += (
            f"\nThe primary airline IATA code to prioritise is '{airline}'. "
            "If there is ambiguity, prefer flights from this airline."
        )

    log_event(logger, "openai_call_started", model=MODEL_NAME)

    try:
        resp = await client.chat.completions.create(
            model=MODEL_NAME,
            max_completion_tokens=2048,
            temperature=0,
            timeout=OPENAI_TIMEOUT_SECONDS,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Extract all flights from this schedule. "
                                "Return ONLY JSON as specified. "
                                "Do NOT include flights that are missing a flight number."
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": image_url},
                        },
                    ],
                },
            ],
        )
    except Exception as e:
        log_event(logger, "openai_call_error", level=logging.ERROR, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

    elapsed = time.time() - t0

    choice = resp.choices[0]
    text = choice.message.content or ""

    usage = resp.usage or {}
    total_tokens = getattr(usage, "total_tokens", None) or 0
    input_tokens = (
        getattr(usage, "prompt_tokens", None)
        or getattr(usage, "input_tokens", None)
        or 0
    )
    output_tokens = (
        getattr(usage, "completion_tokens", None)
        or getattr(usage, "output_tokens", None)
        or 0
    )

    log_event(
        logger,
        "openai_call_finished",
        model=MODEL_NAME,
        duration_ms=int(elapsed * 1000),
        tokens_total=total_tokens,
        tokens_in=input_tokens,
        tokens_out=output_tokens,
    )

    try:
        data = json.loads(text)
    except Exception:
        log_event(logger, "openai_parse_error", level=logging.ERROR, raw_output=text)
        raise HTTPException(400, "Model did not return valid JSON")

    flights = data.get("flights") or []
    connections = data.get("connections") or []

    return {
        "flights": flights,
        "connections": connections,
        "usage": {
            "total_tokens": total_tokens,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        },
    }


# ------------------------------------------------------------------------------
# ROUTES
# ------------------------------------------------------------------------------

@app.get("/health")
async def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "version": app.version,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


@app.post("/extract", response_model=ExtractionResponse)
async def extract(
    request: Request,
    file: UploadFile = File(...),
    airline: Optional[str] = Query(None, description="Airline IATA code, e.g. FX, DL"),
    x_airline: Optional[str] = Header(None, alias="X-Airline"),
):
    """
    Main API pipeline
    """
    overall_start = time.time()

    # 🔥 Merge airline sources and validate
    airline_code = (airline or x_airline or "").strip().upper()
    if not airline_code:
        raise HTTPException(
            status_code=400,
            detail="Airline code required via ?airline=XX or X-Airline header"
        )

    log_event(
        logger,
        "file_processing_started",
        field_filename=file.filename,
        content_type=file.content_type,
        airline=airline_code,
    )

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(400, "Empty file")

    img_stats = _analyse_image_quality(file_bytes)
    log_event(logger, "image_analysis", **img_stats)

    # Step 2: extraction via OpenAI
    page_t0 = time.time()
    log_event(logger, "page_processing_started", page_number=1)

    extraction_raw = await _openai_extract_flights(file_bytes, airline_code)

    page_elapsed = time.time() - page_t0

    # Raw flights straight from model
    flights_raw: List[Dict[str, Any]] = extraction_raw["flights"]

    # Filter out any flights with missing/blank flight_no BEFORE validation
    cleaned_flights: List[Dict[str, Any]] = []
    dropped_missing_flight_no = 0

    for f in flights_raw:
        fn = (f.get("flight_no") or "").strip()
        if not fn:
            dropped_missing_flight_no += 1
            log_event(
                logger,
                "flight_skipped_missing_flight_no",
                raw=f,
            )
            continue
        f["flight_no"] = fn  # normalise whitespace
        cleaned_flights.append(f)

    if dropped_missing_flight_no:
        log_event(
            logger,
            "flight_cleanup_summary",
            total_raw=len(flights_raw),
            kept=len(cleaned_flights),
            dropped_missing_flight_no=dropped_missing_flight_no,
        )

    log_event(
        logger,
        "page_processing_finished",
        page_number=1,
        duration_ms=int(page_elapsed * 1000),
        flights_found=len(cleaned_flights),
    )

    flights: List[Dict[str, Any]] = cleaned_flights
    connections: List[Dict[str, Any]] = extraction_raw["connections"]
    total_flights = len(flights)
    avg_conf = (
        sum(float(f.get("confidence") or 1.0) for f in flights) / total_flights
        if total_flights
        else 0.0
    )

    # 🔥 Filter out flights missing required fields (frontend crashes on nulls)
    complete_flights = [
        f for f in flights
        if f.get("origin") and f.get("dest") 
        and f.get("sched_out_local") and f.get("sched_in_local")
    ]
    
    incomplete_count = len(flights) - len(complete_flights)
    if incomplete_count > 0:
        log_event(
            logger,
            "flights_filtered_incomplete",
            total=len(flights),
            complete=len(complete_flights),
            filtered=incomplete_count,
        )

    extraction_result: Dict[str, Any] = {
        "flights": complete_flights,  # 🔥 Only complete flights
        "connections": connections,
        "total_flights_found": len(complete_flights),  # 🔥 Count only complete
        "avg_conf": avg_conf,  # 🔥 Changed from avg_confidence
        "processing_time": {
            "total_request": time.time() - overall_start,
            "page_1_total": page_elapsed,
        },
        "extraction_method": "direct",
        "metadata": {
            "airline": {"iata": airline_code},
            "file": {
                "name": file.filename,
                "type": file.content_type,
                "size": len(file_bytes),
                "pages": 1,
            },
        },
    }

    # If no valid flights remain, skip validation gracefully
    if len(complete_flights) == 0:
        log_event(
            logger,
            "validation_skipped_no_valid_flights",
        )
        # Keep extraction_result, but still attach cost analysis below
        extraction_validated = extraction_result
    else:
        # Run enrichment + schedule validation
        validation_start = time.time()
        log_event(logger, "validation_started", flights=len(complete_flights))

        extraction_validated = await validate_extraction_results(extraction_result)
        
        # 🔥 Re-filter after validation (enriched_flights might have nulls)
        if "enriched_flights" in extraction_validated:
            complete_enriched = [
                f for f in extraction_validated["enriched_flights"]
                if f.get("origin") and f.get("dest")
                and f.get("sched_out_local") and f.get("sched_in_local")
            ]
            extraction_validated["enriched_flights"] = complete_enriched
            
            log_event(
                logger,
                "enriched_flights_filtered",
                total=len(extraction_validated.get("enriched_flights", [])),
                complete=len(complete_enriched),
            )
        
        # 🔥 Ensure main flights array also has no nulls
        if "flights" in extraction_validated:
            complete_main = [
                f for f in extraction_validated["flights"]
                if f.get("origin") and f.get("dest")
                and f.get("sched_out_local") and f.get("sched_in_local")
            ]
            extraction_validated["flights"] = complete_main
            extraction_validated["total_flights_found"] = len(complete_main)
        
        # 🔥 Fix field name for frontend compatibility
        if "avg_confidence" in extraction_validated:
            extraction_validated["avg_conf"] = extraction_validated.pop("avg_confidence")

        log_event(
            logger,
            "validation_finished",
            duration_ms=int((time.time() - validation_start) * 1000),
        )

    # Cost estimation
    usage = extraction_raw["usage"]
    input_cost = (usage["input_tokens"] / 1_000_000) * 2.50
    output_cost = (usage["output_tokens"] / 1_000_000) * 10.00
    total_cost = input_cost + output_cost

    extraction_validated["cost_analysis"] = {
        "total_tokens": usage["total_tokens"],
        "total_cost_usd": round(total_cost, 4),
        "api_calls": 1,
    }

    log_event(
        logger,
        "http_request_pipeline_completed",
        filename=file.filename,
        airline=airline_code,
        flights_found=extraction_validated.get("total_flights_found", 0),
        duration_ms=int((time.time() - overall_start) * 1000),
    )

    return extraction_validated