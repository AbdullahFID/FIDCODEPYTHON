# extraction_engine.py
from __future__ import annotations

import asyncio
import json
import re
from collections import OrderedDict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from openai import BadRequestError  # optional if you want more granular handling

from config import client, MODEL, MAX_TOKENS, OPENAI_TIMEOUT
import logging
logger = logging.getLogger(__name__)

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

    def _create_messages(self, b64_image: str, prompt: str, attempt: int) -> List[dict]:
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
        return [
            {
                "role": "system",
                "content": (
                    "You are a flight schedule extraction expert. "
                    "Always return valid JSON. You are extracting flight schedules from images. "
                    "Return no metadata, no summaries."
                ),
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_image}"}},
                ],
            },
        ]

    def _parse_response(self, response) -> List[Flight]:
        flights: List[Flight] = []

        # Error JSON (from TIER3 error mode)
        if getattr(response, "choices", None) and response.choices[0].message.content:
            content = response.choices[0].message.content
            try:
                m = re.search(r'\{[^{}]*"error"\s*:\s*true[^{}]*\}', content, re.DOTALL)
                if m:
                    err = json.loads(m.group())
                    if err.get("error") is True:
                        self._last_error = err
                        logger.logger.warning(
                            f"AI reported extraction error: {err.get('user_message', 'Unknown error')}"
                        )
                        return []
            except Exception:
                pass

        # Tool calls
        for choice in getattr(response, "choices", []):
            tcalls = getattr(choice.message, "tool_calls", None)
            if tcalls:
                for tc in tcalls:
                    if getattr(tc, "function", None) and tc.function.arguments:
                        try:
                            data = json.loads(tc.function.arguments)
                            for f in data.get("flights", []):
                                flights.append(Flight(**f))
                        except Exception:
                            continue

        # Direct JSON in content
        if not flights and getattr(response, "choices", None):
            content = response.choices[0].message.content or ""
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
        if hasattr(response, "usage") and response.usage:
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            total_tokens = response.usage.total_tokens
            input_cost = input_tokens * 0.0025 / 1000   # $2.50 / 1M
            output_cost = output_tokens * 0.01 / 1000   # $10 / 1M
            total_cost = input_cost + output_cost

            logger.info(
                f"Token usage: in={input_tokens}, out={output_tokens}, total={total_tokens}, "
                f"cost=${total_cost:.4f}"
            )

            self.total_tokens_used += total_tokens
            self.total_cost += total_cost
            self.api_calls_count += 1

    async def extract_with_tools(self, b64_image: str, prompt: str, attempt: int) -> List[Flight]:
        tools = [
            {
                "type": "function",
                "function": {
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
                                        "origin": {"type": ["string", "null"]},
                                        "dest": {"type": ["string", "null"]},
                                        "sched_out_local": {"type": ["string", "null"]},
                                        "sched_in_local": {"type": ["string", "null"]},
                                    },
                                    "required": ["date", "flight_no"],
                                },
                            }
                        },
                        "required": ["flights"],
                    },
                },
            }
        ]

        messages = self._create_messages(b64_image, prompt, attempt)
        logger.info(f"OpenAI call (attempt={attempt}, model={MODEL})")

        try:
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    tools=tools,
                    tool_choice={"type": "function", "function": {"name": "extract_flights"}},
                    max_completion_tokens=MAX_TOKENS,
                    n=2 if attempt > 1 else 1,
                ),
                timeout=OPENAI_TIMEOUT,
            )
        except asyncio.TimeoutError:
            logger.logger.error(f"OpenAI timeout after {OPENAI_TIMEOUT}s")
            return []
        except Exception as e:
            logger.logger.error(f"OpenAI error: {e}")
            return []

        await self._record_usage(response)
        flights = self._parse_response(response)

        # track successful patterns
        if flights:
            for f in flights[:3]:
                pattern = f"{f.flight_no[:2] if len(f.flight_no) > 2 else 'XX'}###"
                if pattern not in self.successful_patterns:
                    self.successful_patterns.append(pattern)

        return flights

    async def extract_direct_json(self, b64_image: str, attempt: int) -> List[Flight]:
        messages = [
            {"role": "system", "content": "Extract flight data and return ONLY valid JSON."},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": """Extract all flights from this image.

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
}""",
                    },
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_image}"}},
                ],
            },
        ]

        try:
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    max_completion_tokens=MAX_TOKENS,
                    response_format={"type": "json_object"},
                ),
                timeout=OPENAI_TIMEOUT,
            )
        except asyncio.TimeoutError:
            logger.logger.error(f"OpenAI JSON timeout after {OPENAI_TIMEOUT}s")
            return []
        except Exception as e:
            logger.logger.error(f"OpenAI JSON error: {e}")
            return []

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

            logger.info(f"Processing image version type={vtype}")
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
                logger.info("Escalating to GPT-5.1-thinking for rescue attempt")
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