# extraction_engine.py
from __future__ import annotations

import asyncio
import json
import re
from collections import OrderedDict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from config import MODEL, MAX_TOKENS, TIMEOUT
from logging_utils import get_logger
logger = get_logger("extraction_engine")
from models import Flight
from patterns import patterns
from prompts import (
    TIER1_STRUCTURED_PROMPT,
    TIER2_AGGRESSIVE_PROMPT,
    TIER3_FORENSIC_PROMPT,
)


class PerfectExtractionEngine:
    """OpenAI-based extractor with multi-tier prompts and tool calls."""

    def __init__(self) -> None:
        self.successful_patterns: List[str] = []
        self._last_error: Optional[Dict[str, Any]] = None
        self.total_tokens_used: int = 0
        self.total_cost: float = 0.0
        self.api_calls_count: int = 0

    # ---------------- core helpers ----------------

    def _create_content(self, b64_image: str, prompt: str, attempt: int) -> List[Any]:
        mobile_hint = """
CRITICAL: If you see a calendar grid with:
- Colored bars containing 4-digit numbers (like 9013, 8619)
- Airport codes next to dates (MCO, LAS, TPA)
- These are FLIGHTS! Extract them!
"""
        if attempt == 1:
            prompt = mobile_hint + "\n\n" + prompt
        if self.successful_patterns and attempt > 1:
            prompt += f"\n\nPreviously successful patterns: {', '.join(self.successful_patterns[:3])}"
        
        # Gemini accepts PIL images or bytes. Since we have b64, we convert back to PIL/Bytes.
        import base64
        import io
        from PIL import Image
        
        image_bytes = base64.b64decode(b64_image)
        img = Image.open(io.BytesIO(image_bytes))
        
        system_instruction = (
            "You are a flight schedule extraction expert. "
            "Always return valid JSON. You are extracting flight schedules from images. "
            "Return no metadata, no summaries."
        )
        
        # In Gemini, system instruction is set on the model, but we can prepend it to prompt 
        # or use the system_instruction argument when creating the model.
        # For simplicity in this method, we return the parts for the user message.
        
        return [prompt, img]

    def _parse_response(self, response) -> List[Flight]:
        flights: List[Flight] = []

        # Gemini response structure
        # response.parts might contain text or function calls
        
        try:
            for part in response.parts:
                if fn := part.function_call:
                    if fn.name == "extract_flights":
                        # Convert MapComposite to dict
                        args = dict(fn.args)
                        # args['flights'] might be a list of MapComposite
                        raw_flights = args.get("flights", [])
                        for f in raw_flights:
                            # Convert each flight to dict if it's not already
                            f_dict = dict(f) if hasattr(f, "items") else f
                            try:
                                flights.append(Flight(**f_dict))
                            except Exception:
                                continue
        except Exception:
            pass

        # Direct JSON in text
        if not flights and response.text:
            content = response.text
            json_patterns = [
                r"```json\s*([\s\S]*?)\s*```",
                r"```\s*([\s\S]*?)\s*```",
                r"(\{[\s\S]*\})",
            ]
            for pat in json_patterns:
                for match in re.findall(pat, content):
                    try:
                        cleaned = match.strip()
                        if cleaned.startswith("json"):
                            cleaned = cleaned[4:].strip()
                        data = json.loads(cleaned)
                        if isinstance(data, dict) and "flights" in data:
                            for f in data["flights"]:
                                flights.append(Flight(**f))
                    except Exception:
                        continue

            # very crude fallback: synthetic flights from text hints
            if not flights and "calendar" in content.lower():
                for fn in re.findall(r"\b([89]\d{3})\b", content):
                    flights.append(
                        Flight(
                            flight_no=fn,
                            date=datetime.now().strftime("%m/%d/%Y"),
                            confidence=0.7,
                        )
                    )
            if not flights:
                flights.extend(self._extract_from_text(content))

        return flights

    def _extract_from_text(self, text: str) -> List[Flight]:
        out: List[Flight] = []
        for line in text.splitlines():
            m = patterns.FLIGHT_NO.search(line)
            if not m:
                continue
            data: Dict[str, Any] = {"flight_no": m.group()}
            airports = patterns.AIRPORT.findall(line)
            if len(airports) >= 2:
                data["origin"], data["dest"] = airports[0], airports[1]
            dm = patterns.DATE_DMY.search(line) or patterns.DATE_MDY.search(line)
            if dm:
                data["date"] = self._parse_date(dm)
            times = patterns.TIME_24H.findall(line)
            if times:
                if len(times) >= 1:
                    data["sched_out_local"] = f"{times[0][0]}{times[0][1]}"
                if len(times) >= 2:
                    data["sched_in_local"] = f"{times[1][0]}{times[1][1]}"
            if "date" not in data:
                data["date"] = datetime.now().strftime("%m/%d/%Y")
            try:
                out.append(Flight(**data))
            except Exception:
                continue
        return out

    def _parse_date(self, match) -> str:
        """
        Parse date with intelligent year inference for 2026+ dates.
        
        ✅ FIXED: Now handles all months in 2026, not just Jan-Feb!
        
        Strategy:
        1. If month < current_month → next year (forward-looking schedules)
        2. If month >= current_month → current year
        3. Cap at 18 months forward to prevent far-future errors
        4. Always respect explicit years
        """
        txt = match.group(0)
        now = datetime.now()
        current_year = now.year
        current_month = now.month
        
        # ════════════════════════════════════════════════════════════
        # PATTERN 1: DD/Mon format (e.g., "24/Aug")
        # ════════════════════════════════════════════════════════════
        m = re.match(r"(\d{1,2})/([A-Za-z]{3})", txt)
        if m:
            day = int(m.group(1))
            mon_str = m.group(2).lower()
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
            month = month_map.get(mon_str, current_month)
            
            # ✅ FIX: Any past month → next year
            # Example: Dec 2025 parsing "Sept" → Sept 2026 (not Sept 2025)
            year = current_year + (1 if month < current_month else 0)
            
            # Cap at 18 months forward to prevent far-future misinterpretation
            months_forward = (year - current_year) * 12 + (month - current_month)
            if months_forward > 18:
                year = current_year
            
            return f"{month:02d}/{day:02d}/{year}"
        
        # ════════════════════════════════════════════════════════════
        # PATTERN 2: Numeric format (MM/DD or DD/MM with optional year)
        # ════════════════════════════════════════════════════════════
        m = re.match(r"(\d{1,2})[/-](\d{1,2})(?:[/-](\d{2,4}))?", txt)
        if m:
            a, b, y = m.group(1), m.group(2), m.group(3)
            a_i, b_i = int(a), int(b)
            
            # Handle explicit year if provided
            if y:
                year = int(y) if len(y) == 4 else 2000 + int(y)
            else:
                year = current_year
            
            # Determine MM/DD vs DD/MM format
            if 1 <= a_i <= 12 and 1 <= b_i <= 31:
                month, day = a_i, b_i  # MM/DD format
            else:
                month, day = b_i, a_i  # DD/MM format
            
            # ✅ FIX: Any past month → next year (only if year wasn't explicit)
            if not y:
                year = current_year + (1 if month < current_month else 0)
                
                # Cap at 18 months forward
                months_forward = (year - current_year) * 12 + (month - current_month)
                if months_forward > 18:
                    year = current_year
            
            return f"{month:02d}/{day:02d}/{year}"
        
        # ════════════════════════════════════════════════════════════
        # FALLBACK: Use current date
        # ════════════════════════════════════════════════════════════
        return now.strftime("%m/%d/%Y")

    # ---------------- OpenAI calls ----------------

    async def _record_usage(self, response) -> None:
        if hasattr(response, "usage_metadata"):
            usage = response.usage_metadata
            input_tokens = usage.prompt_token_count
            output_tokens = usage.candidates_token_count
            total_tokens = usage.total_token_count
            # Gemini Flash pricing approx
            input_cost = input_tokens * 0.10 / 1_000_000
            output_cost = output_tokens * 0.40 / 1_000_000
            total_cost = input_cost + output_cost

            logger.logger.info(
                f"Token usage: in={input_tokens}, out={output_tokens}, total={total_tokens}, "
                f"cost=${total_cost:.4f}"
            )

            self.total_tokens_used += total_tokens
            self.total_cost += total_cost
            self.api_calls_count += 1

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True
    )
    async def extract_with_tools(self, b64_image: str, prompt: str, attempt: int) -> List[Flight]:
        # Define the tool structure for Gemini
        extract_flights_tool = {
            "function_declarations": [
                {
                    "name": "extract_flights",
                    "description": "Extract all flight information from the image",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "flights": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "date": {"type": "string"},
                                        "flight_no": {"type": "string"},
                                        "origin": {"type": "string"},
                                        "dest": {"type": "string"},
                                        "sched_out_local": {"type": "string"},
                                        "sched_in_local": {"type": "string"},
                                    },
                                    "required": ["date", "flight_no"],
                                },
                            }
                        },
                        "required": ["flights"],
                    },
                }
            ]
        }

        content = self._create_content(b64_image, prompt, attempt)
        logger.logger.info(f"Gemini call (attempt={attempt}, model={MODEL})")

        model = genai.GenerativeModel(
            MODEL, 
            tools=[extract_flights_tool],
            system_instruction="You are a flight schedule extraction expert. Always return valid JSON."
        )

        # Gemini async generation
        req_time = datetime.now().isoformat()
        logger.logger.info(f"[REQUEST_TRACKER] Sending request to Gemini (model={MODEL}) at {req_time} | Attempt: {attempt}")
        
        response = await model.generate_content_async(
            content,
            generation_config=genai.types.GenerationConfig(
                temperature=0.1,
                # Force function calling if possible, or auto
                tool_config={'function_calling_config': {'mode': 'ANY'}}
            )
        )

        await self._record_usage(response)
        flights = self._parse_response(response)

        # track successful patterns
        if flights:
            for f in flights[:3]:
                pattern = f"{f.flight_no[:2] if len(f.flight_no) > 2 else 'XX'}###"
                if pattern not in self.successful_patterns:
                    self.successful_patterns.append(pattern)

        return flights

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True
    )
    async def extract_direct_json(self, b64_image: str, attempt: int) -> List[Flight]:
        content = self._create_content(b64_image, """Extract all flights from this image.

Return ONLY this JSON structure:
{
  "flights": [
    {
      "date": "MM/DD/YYYY",
      "flight_no": "####",
      "origin": "XXX",
      "dest": "XXX",
      "sched_out_local": "HHMM",
      "sched_in_local": "HHMM"
    }
  ]
}""", attempt)

        model = genai.GenerativeModel(
            MODEL,
            system_instruction="Extract flight data and return ONLY valid JSON."
        )

        req_time = datetime.now().isoformat()
        logger.logger.info(f"[REQUEST_TRACKER] Sending request to Gemini (model={MODEL}) at {req_time} | Attempt: {attempt} | Mode: JSON")

        response = await model.generate_content_async(
            content,
            generation_config=genai.types.GenerationConfig(
                temperature=0.1,
                response_mime_type="application/json"
            )
        )

        await self._record_usage(response)
        return self._parse_response(response)

    # ---------------- multi-strategy orchestrator ----------------

    async def extract_comprehensive(
        self,
        image_versions: List[Tuple[str, str]],
        *,
        stop_on_first_success: bool = True,
        min_flights: int = 1,
    ) -> List[Flight]:
        """Run through TIER1/2/3 across all image variants with early-exit."""

        self.total_tokens_used = 0
        self.total_cost = 0.0
        self.api_calls_count = 0
        self._last_error = None

        all_flights: List[Flight] = []
        seen: set[str] = set()

        for version_idx, (b64_image, vtype) in enumerate(image_versions):
            if stop_on_first_success and all_flights:
                return all_flights

            logger.logger.info(f"Processing image version type={vtype}")
            if vtype == "original":
                prompts = [TIER1_STRUCTURED_PROMPT, TIER2_AGGRESSIVE_PROMPT]
            elif vtype == "enhanced":
                prompts = [TIER2_AGGRESSIVE_PROMPT, TIER1_STRUCTURED_PROMPT]
            else:
                prompts = [TIER3_FORENSIC_PROMPT]

            for attempt, prompt in enumerate(prompts, 1):
                op = f"extract_{vtype}_attempt_{attempt}"
                logger.start_timer(op)
                flights = await self.extract_with_tools(b64_image, prompt, attempt)
                logger.end_timer(op)

                if (not flights) and (attempt == len(prompts)) and (not all_flights):
                    opj = f"extract_{vtype}_json"
                    logger.start_timer(opj)
                    flights = await self.extract_direct_json(b64_image, attempt)
                    logger.end_timer(opj)

                for f in flights:
                    key = f"{f.flight_no}_{f.date}"
                    if key not in seen:
                        seen.add(key)
                        f.confidence = max(0.0, 1.0 - (0.1 * (version_idx + attempt - 1)))
                        all_flights.append(f)

                logger.log_extraction(len(flights), attempt, f"{vtype}_tier")

                if flights and stop_on_first_success and len(all_flights) >= min_flights:
                    return all_flights

                if flights:
                    break

        # optional GPT-5.1-thinking rescue path
        if not all_flights and image_versions:
            try:
                original_model = MODEL
                logger.logger.info("Escalating to Gemini rescue attempt")
                # You could swap model via env or config here; for now we just reuse MODEL
                flights = await self.extract_with_tools(
                    image_versions[0][0],
                    TIER3_FORENSIC_PROMPT,
                    attempt=999,
                )
                all_flights.extend(flights)
            except Exception as e:
                logger.logger.error(f"Rescue attempt failed: {e}")

        return all_flights

    @property
    def extraction_error(self) -> Optional[Dict[str, Any]]:
        return self._last_error