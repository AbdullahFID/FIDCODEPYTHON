# api.py
import base64
import time
import re
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, Header, HTTPException, Query, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field
from zoneinfo import ZoneInfo

from airlines import AIRLINE_CODES
from flight_intel_patch.validator import validate_extraction_results
from logging_utils import logger
from models import Result
from pipeline import UltimatePipeline
from pdf_processor import PDFProcessor
import functools as _functools

# ---------------- airline resolver ----------------


def resolve_airline(code: str) -> dict:
    """
    Permissive resolver:
      • Known IATA in AIRLINE_CODES  -> return with 'iata'
      • Known ICAO in AIRLINE_CODES  -> return mapped to canonical 'iata'
      • Unknown but syntactically valid IATA/ICAO -> accept without metadata
      • Invalid format -> 400
    """
    from fastapi import HTTPException

    code = (code or "").strip().upper()

    # Known IATA
    if code in AIRLINE_CODES:
        rec = AIRLINE_CODES[code].copy()
        rec.setdefault("iata", code)
        return rec

    # Known ICAO
    for iata, data in AIRLINE_CODES.items():
        if data.get("icao", "").upper() == code:
            rec = data.copy()
            rec.setdefault("iata", iata)
            return rec

    # Unknown but syntactically valid
    if re.fullmatch(r"[A-Z0-9]{2,3}", code):
        if len(code) == 2:
            return {"iata": code, "icao": None, "name": None, "country": None}
        return {"iata": None, "icao": code, "name": None, "country": None}

    raise HTTPException(400, f"Airline code '{code}' is not a valid IATA/ICAO format")


# ---------------- date normalizer ----------------


def normalize_dates(flights: List[Dict[str, Any]]) -> None:
    today = datetime.now(ZoneInfo("America/Toronto"))
    current_year = today.year
    current_month = today.month
    month_map = {
        "jan": 1,
        "feb": 2,
        "mar": 3,
        "apr": 4,
        "may": 5,
        "jun": 6,
        "jul": 7,
        "aug": 8,
        "sep": 9,
        "oct": 10,
        "nov": 11,
        "dec": 12,
    }

    for f in flights:
        date_str = f.get("date", "")
        if not date_str:
            continue

        if "/" in date_str and any(mon in date_str.lower() for mon in month_map):
            parts = date_str.split("/")
            if len(parts) == 2:
                day = parts[0].strip()
                mon_str = parts[1].strip().lower()[:3]
                if mon_str in month_map:
                    month = month_map[mon_str]
                    year = current_year + (1 if current_month >= 10 and month <= 2 else 0)
                    f["date"] = f"{month:02d}/{day.zfill(2)}/{year}"
                    continue

        if "/" in date_str:
            parts = date_str.split("/")
            if len(parts) >= 2:
                try:
                    month = int(parts[0])
                    day = int(parts[1])
                    year = int(parts[2]) if len(parts) > 2 else current_year
                    if year < 100:
                        year += 2000
                    if current_month >= 10 and month <= 2:
                        year = current_year + 1
                    elif year < current_year:
                        year = current_year
                    f["date"] = f"{month:02d}/{day:02d}/{year}"
                except Exception:
                    continue


# ---------------- FastAPI app ----------------

pipeline = UltimatePipeline()

app = FastAPI(
    title="Flight-Intel v8.x",
    version="8.1.0",
    description="Maximum accuracy flight extraction with GPT-5.1",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)
app.add_middleware(GZipMiddleware, minimum_size=1000)


@app.post("/extract", response_model=None)
async def extract_flights(
    file: UploadFile = File(..., description="Image or PDF file"),
    airline: Optional[str] = Query(None, description="Airline code (IATA/ICAO)"),
    x_airline: Optional[str] = Header(None, alias="X-Airline"),
):
    logger.start_timer("http_request")
    request_start = time.perf_counter()

    try:
        ctype = (file.content_type or "").lower()
        is_pdf = ctype == "application/pdf"
        is_image = ctype.startswith("image/")
        if not (is_pdf or is_image):
            raise HTTPException(415, "Please upload an image (PNG/JPEG) or PDF file")

        airline_code = (airline or x_airline or "").strip()
        if not airline_code:
            raise HTTPException(
                400, "Please provide airline code via ?airline=XX or X-Airline header"
            )

        airline_info = resolve_airline(airline_code)
        iata_prefix = airline_info.get("iata") or ""

        logger.start_timer("file_processing")
        file_data = await file.read()

        if is_pdf:
            logger.logger.info(
                f"Processing PDF: {file.filename} ({len(file_data):,} bytes)"
            )
            images = await PDFProcessor.convert(file_data)
        else:
            np_img = cv2.imdecode(np.frombuffer(file_data, np.uint8), cv2.IMREAD_COLOR)
            if np_img is None:
                raise HTTPException(422, "Unable to decode image file")
            images = [np_img]
            logger.logger.info(f"Processing image: {file.filename}")

        logger.end_timer("file_processing")

        result: Result = await pipeline.process(images)

        # propagate TIER3-style error if extraction failed
        if pipeline.extractor.extraction_error and not result.flights:
            err = pipeline.extractor.extraction_error
            return {
                "success": False,
                "error": True,
                "message": err.get(
                    "user_message", "Unable to extract flight information from the image"
                ),
                "technical_reason": err.get("technical_reason", "Unknown extraction failure"),
                "suggestions": err.get(
                    "suggestions",
                    [
                        "Ensure the image is clear and well-lit",
                        "Make sure the entire schedule is visible",
                        "Try taking the photo from directly above",
                    ],
                ),
                "metadata": {
                    "airline": airline_info,
                    "file": {
                        "name": file.filename,
                        "type": "pdf" if is_pdf else "image",
                        "size": len(file_data),
                    },
                    "timestamp": datetime.now().isoformat(),
                },
            }

        # prefix numeric flight numbers with airline
        for flight in result.flights:
            if not flight.flight_no or not iata_prefix:
                continue
            fn = flight.flight_no
            if fn.isdigit():
                flight.flight_no = f"{iata_prefix}{fn}"
            elif fn[0].isdigit() and not fn.upper().startswith(iata_prefix):
                flight.flight_no = f"{iata_prefix}{fn}"

        output = jsonable_encoder(result)
        normalize_dates(output.get("flights", []))

        output["metadata"] = {
            "airline": airline_info,
            "file": {
                "name": file.filename,
                "type": "pdf" if is_pdf else "image",
                "size": len(file_data),
                "pages": len(images) if is_pdf else 1,
            },
            "processing": {
                "started": datetime.fromtimestamp(request_start).isoformat(),
                "version": "8.1.0",
            },
        }

        # do we need validation enrichment?
        needs_enrichment = any(
            not all(
                [
                    f.get("origin"),
                    f.get("dest"),
                    f.get("sched_out_local"),
                    f.get("sched_in_local"),
                ]
            )
            for f in output.get("flights", [])
        )

        if needs_enrichment and output.get("flights"):
            logger.logger.info("Validation needed - missing fields detected")

            miss = defaultdict(int)
            for f in output.get("flights", []):
                if not f.get("origin"):
                    miss["origin"] += 1
                if not f.get("dest"):
                    miss["dest"] += 1
                if not f.get("sched_out_local"):
                    miss["sched_out_local"] += 1
                if not f.get("sched_in_local"):
                    miss["sched_in_local"] += 1
            logger.logger.info(f"Missing fields summary: {dict(miss)}")

            logger.start_timer("validation")
            try:
                enriched = await validate_extraction_results(output)
                output.update(enriched)
            finally:
                logger.end_timer("validation")
        else:
            logger.logger.info("All fields complete - skipping validation")

        output.setdefault("processing_time", {})
        output["processing_time"]["total_request"] = logger.end_timer("http_request")

        # cost summary from extractor
        if pipeline.extractor.total_tokens_used:
            output["cost_analysis"] = {
                "total_tokens": pipeline.extractor.total_tokens_used,
                "total_cost_usd": round(pipeline.extractor.total_cost, 4),
                "api_calls": pipeline.extractor.api_calls_count,
                "avg_tokens_per_call": (
                    pipeline.extractor.total_tokens_used
                    // max(pipeline.extractor.api_calls_count, 1)
                ),
                "pricing_model": "gpt-5.1 ($2.50/1M input, $10/1M output)",
            }

        return output

    except HTTPException:
        logger.end_timer("http_request")
        raise
    except Exception as e:
        logger.end_timer("http_request")
        logger.logger.error(f"Request failed: {e}", exc_info=True)
        raise HTTPException(500, f"Processing failed: {str(e)}")


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "version": "8.1.0",
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/")
async def root():
    return {
        "name": "Flight-Intel v8.1",
        "description": "Maximum accuracy flight schedule extraction",
        "endpoints": {
            "/extract": "POST - Extract flights from image/PDF",
            "/health": "GET - System health status",
            "/docs": "GET - Interactive API documentation",
        },
        "version": "8.1.0",
    }

