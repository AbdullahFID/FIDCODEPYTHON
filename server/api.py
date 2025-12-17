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
import cv2
import numpy as np

import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from config import MODEL, TIMEOUT, MAX_WORKERS
from logging_utils import configure_logging, log_event, new_request_id
from flight_intel_patch.validator import validate_extraction_results
from pdf_processor import PDFProcessor

# ------------------------------------------------------------------------------
# APP + LOGGING SETUP
# ------------------------------------------------------------------------------

configure_logging()
logger = logging.getLogger("flightintel.api")

app = FastAPI(title="Flight-Intel v8.1", version="8.1.0")

# Basic CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8001", "http://127.0.0.1:8001", "http://0.0.0.0:8001"],
    allow_origin_regex=r"http://localhost:\d+",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger.info(
    "Config: model=%s, timeout=%ss, workers=%s",
    MODEL,
    TIMEOUT,
    MAX_WORKERS,
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


# 🔥 NEW: DETECT IF FILE IS PDF
def _is_pdf(file_bytes: bytes, content_type: Optional[str]) -> bool:
    """
    Check if file is PDF via MIME type or magic bytes.
    """
    # Check MIME type
    if content_type and "pdf" in content_type.lower():
        return True
    
    # Check magic bytes (PDF starts with %PDF)
    if file_bytes[:4] == b'%PDF':
        return True
    
    return False


# 🔥 NEW: CONVERT CV2 IMAGE TO BYTES
def _cv2_to_bytes(cv_img: np.ndarray) -> bytes:
    """
    Convert OpenCV image (np.ndarray) to bytes for downstream processing.
    """
    success, buffer = cv2.imencode('.png', cv_img)
    if not success:
        raise ValueError("Failed to encode image to PNG")
    return buffer.tobytes()


# ------------------------------------------------------------------------------
# OPENAI VISION (CORRECTED + STRONG CONSTRAINTS)
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# GEMINI VISION
# ------------------------------------------------------------------------------

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(Exception),
    reraise=True
)
async def _gemini_extract_flights(
    image_bytes: bytes,
    airline: Optional[str],
    page_num: int = 1,
) -> Dict[str, Any]:

    import json

    t0 = time.time()

    # Create Gemini model instance
    model = genai.GenerativeModel(MODEL)

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
        f'      \"page_number\": {page_num},\n'
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

    log_event(logger, "gemini_call_started", model=MODEL, page=page_num)

    # Create image part
    # Gemini accepts bytes directly for some formats, or we can use PIL
    img = Image.open(io.BytesIO(image_bytes))
    
    prompt = (
        "Extract all flights from this schedule. "
        "Return ONLY JSON as specified. "
        "Do NOT include flights that are missing a flight number."
    )

    # Generate content
    # Note: genai.GenerativeModel.generate_content_async is available in newer versions
    # or we can run in executor if async is not fully supported yet.
    # Assuming standard google-generativeai usage.
    
    req_time = datetime.now().isoformat()
    logger.info(f"[REQUEST_TRACKER] Sending request to Gemini (model={MODEL}) at {req_time} | Page: {page_num}")

    response = await model.generate_content_async(
        [system_prompt, prompt, img],
        generation_config=genai.types.GenerationConfig(
            temperature=0,
            response_mime_type="application/json"
        )
    )

    elapsed = time.time() - t0
    text = response.text or ""

    # Usage metadata might be available in response.usage_metadata
    usage = response.usage_metadata
    total_tokens = usage.total_token_count if usage else 0
    input_tokens = usage.prompt_token_count if usage else 0
    output_tokens = usage.candidates_token_count if usage else 0

    log_event(
        logger,
        "gemini_call_finished",
        model=MODEL,
        page=page_num,
        duration_ms=int(elapsed * 1000),
        tokens_total=total_tokens,
        tokens_in=input_tokens,
        tokens_out=output_tokens,
    )

    try:
        data = json.loads(text)
    except Exception:
        # Try to clean markdown
        try:
            cleaned = text.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            data = json.loads(cleaned)
        except Exception:
            log_event(logger, "gemini_parse_error", level=logging.ERROR, raw_output=text, page=page_num)
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
    Main API pipeline - now supports both images AND PDFs! 🎉
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

    # 🔥 PDF DETECTION AND CONVERSION
    is_pdf_file = _is_pdf(file_bytes, file.content_type)
    
    if is_pdf_file:
        log_event(logger, "pdf_detected", filename=file.filename)
        
        # Convert PDF to list of cv2 images
        try:
            cv_images = await PDFProcessor.convert(file_bytes)
            log_event(logger, "pdf_converted", pages=len(cv_images))
        except Exception as e:
            log_event(logger, "pdf_conversion_failed", level=logging.ERROR, error=str(e))
            raise HTTPException(422, f"PDF conversion failed: {str(e)}")
        
        # Process each page
        all_flights: List[Dict[str, Any]] = []
        all_connections: List[Dict[str, Any]] = []
        total_usage = {"total_tokens": 0, "input_tokens": 0, "output_tokens": 0}
        page_times: Dict[str, float] = {}
        
        for page_idx, cv_img in enumerate(cv_images, start=1):
            page_t0 = time.time()
            log_event(logger, "page_processing_started", page_number=page_idx)
            
            # Convert cv2 image back to bytes for processing
            page_bytes = _cv2_to_bytes(cv_img)
            
            # Run image quality analysis on first page only (for metadata)
            if page_idx == 1:
                img_stats = _analyse_image_quality(page_bytes)
                log_event(logger, "image_analysis", page=page_idx, **img_stats)
            
            # Extract flights from this page
            extraction_raw = await _gemini_extract_flights(page_bytes, airline_code, page_idx)
            
            page_elapsed = time.time() - page_t0
            page_times[f"page_{page_idx}_total"] = page_elapsed
            
            # Accumulate flights
            for flight in extraction_raw["flights"]:
                flight["page_number"] = page_idx  # Tag with page number
                all_flights.append(flight)
            
            all_connections.extend(extraction_raw["connections"])
            
            # Accumulate token usage
            for key in ["total_tokens", "input_tokens", "output_tokens"]:
                total_usage[key] += extraction_raw["usage"].get(key, 0)
            
            log_event(
                logger,
                "page_processing_finished",
                page_number=page_idx,
                duration_ms=int(page_elapsed * 1000),
                flights_found=len(extraction_raw["flights"]),
            )
        
        # Use accumulated results
        flights_raw = all_flights
        connections = all_connections
        usage = total_usage
        processing_time = page_times
        processing_time["total_request"] = time.time() - overall_start
        total_pages = len(cv_images)
        
    else:
        # 🔥 STANDARD IMAGE PROCESSING (original flow)
        log_event(logger, "image_detected", filename=file.filename)
        
        img_stats = _analyse_image_quality(file_bytes)
        log_event(logger, "image_analysis", **img_stats)

        # Step 2: extraction via OpenAI
        page_t0 = time.time()
        log_event(logger, "page_processing_started", page_number=1)

        extraction_raw = await _gemini_extract_flights(file_bytes, airline_code, page_num=1)

        page_elapsed = time.time() - page_t0

        flights_raw = extraction_raw["flights"]
        connections = extraction_raw["connections"]
        usage = extraction_raw["usage"]
        processing_time = {
            "total_request": time.time() - overall_start,
            "page_1_total": page_elapsed,
        }
        total_pages = 1
        
        log_event(
            logger,
            "page_processing_finished",
            page_number=1,
            duration_ms=int(page_elapsed * 1000),
            flights_found=len(flights_raw),
        )

    # 🔥 COMMON PROCESSING (same for images and PDFs)
    
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

    flights: List[Dict[str, Any]] = cleaned_flights
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
        "processing_time": processing_time,
        "extraction_method": "pdf" if is_pdf_file else "direct",  # 🔥 Track method
        "metadata": {
            "airline": {"iata": airline_code},
            "file": {
                "name": file.filename,
                "type": file.content_type,
                "size": len(file_bytes),
                "pages": total_pages,  # 🔥 Accurate page count
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

    # Cost estimation (Gemini 2.0 Flash pricing: Free tier or low cost)
    # Using approx $0.10/1M input, $0.40/1M output for Flash (illustrative)
    input_cost = (usage["input_tokens"] / 1_000_000) * 0.10
    output_cost = (usage["output_tokens"] / 1_000_000) * 0.40
    total_cost = input_cost + output_cost

    extraction_validated["cost_analysis"] = {
        "total_tokens": usage["total_tokens"],
        "total_cost_usd": round(total_cost, 4),
        "api_calls": total_pages,  # 🔥 One call per page
    }

    log_event(
        logger,
        "http_request_pipeline_completed",
        filename=file.filename,
        airline=airline_code,
        is_pdf=is_pdf_file,
        pages=total_pages,
        flights_found=extraction_validated.get("total_flights_found", 0),
        duration_ms=int((time.time() - overall_start) * 1000),
    )

    return extraction_validated