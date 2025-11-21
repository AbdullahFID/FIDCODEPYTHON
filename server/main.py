# main.py
# Flight-Intel v8.0 ULTIMATE — optimized & cleaned (no prompt changes)

from dotenv import load_dotenv
load_dotenv()

import os
import re
import cv2
import json
import time
import base64
import asyncio
import logging
import hashlib
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any
from collections import defaultdict, OrderedDict
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field, field_validator  # 🔥 Add field_validator
from pdf2image import convert_from_bytes
from zoneinfo import ZoneInfo
from openai import AsyncOpenAI
import functools as _functools

# External helpers
from flight_intel_patch import validate_extraction_results
from airlines import AIRLINE_CODES

# ================= Logging =================
class PerfectLogger:
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.timers: Dict[str, float] = {}
        self.stats: Dict[str, List[float]] = defaultdict(list)

    def start_timer(self, operation: str) -> float:
        t = time.perf_counter()
        self.timers[operation] = t
        self.logger.info(f"⏱️  [{operation}] Started")
        return t

    def end_timer(self, operation: str) -> float:
        if operation not in self.timers:
            return 0.0
        elapsed = time.perf_counter() - self.timers.pop(operation)
        self.stats[operation].append(elapsed)
        emoji = "⚡" if elapsed < 1 else "✅" if elapsed < 3 else "⏰"
        self.logger.info(f"{emoji} [{operation}] Completed in {elapsed:.2f}s")
        return elapsed

    def log_extraction(self, flights_count: int, attempt: int, method: str):
        if flights_count > 0:
            self.logger.info(f"✈️  Extracted {flights_count} flights on attempt {attempt} using {method}")
        else:
            self.logger.warning(f"⚠️  No flights found on attempt {attempt} using {method}")

    def log_api_call(self, api_name: str, params: Dict, response: Any = None):
        self.logger.info(f"🌐 API CALL: {api_name}")
        self.logger.info(f"   📤 Params: {json.dumps(params, default=str)[:500]}")
        if response:
            self.logger.info(f"   📥 Response: {json.dumps(response, default=str)[:500]}")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S")
logger = PerfectLogger("flight-intel")

# ================= Config =================
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_KEY:
    raise RuntimeError("❌ OPENAI_API_KEY missing")

MODEL = "gpt-5.1"
MAX_TOKENS = 4096
OPENAI_TIMEOUT = 30

MAX_WORKERS = min(32, (os.cpu_count() or 4) * 2)
thread_pool = ThreadPoolExecutor(max_workers=MAX_WORKERS)
client = AsyncOpenAI(api_key=OPENAI_KEY)

# ================= Regex =================
class Patterns:
    FLIGHT_NO = re.compile(r"\b(?:[A-Z]{1,2}\d{1,5}[A-Z]?|[A-Z]\d{5}[A-Z]?|\d{3,5})\b")
    DATE_DMY = re.compile(r"\b(\d{1,2})[/-](\d{1,2})(?:[/-](\d{2,4}))?\b")
    DATE_MDY = re.compile(r"\b(\d{1,2})/([A-Za-z]{3})\b")
    TIME_24H = re.compile(r"\b([01]?\d|2[0-3]):?([0-5]\d)\b")
    AIRPORT = re.compile(r"\b[A-Z]{3}\b")
    ROUTE = re.compile(r"\b([A-Z]{3})\s*[-–→>]\s*([A-Z]{3})\b")
patterns = Patterns()

# ================= Models =================
class Flight(BaseModel):
    date: str = Field(..., description="MM/DD/YYYY")
    flight_no: str = Field(..., description="Flight number")
    origin: Optional[str] = None
    dest: Optional[str] = None
    sched_out_local: Optional[str] = None
    sched_in_local: Optional[str] = None
    page_number: Optional[int] = None
    confidence: float = Field(default=1.0, ge=0, le=1)

    @field_validator("flight_no")  # 🔥 Changed from @validator
    @classmethod  # 🔥 Added @classmethod
    def _clean_flight_no(cls, v: str) -> str:
        if not v:
            return v
        return re.sub(r"[^\w\d]", "", v.upper())

    @field_validator("origin", "dest")  # 🔥 Changed from @validator
    @classmethod  # 🔥 Added @classmethod
    def _validate_airport(cls, v: Optional[str]) -> Optional[str]:
        return v.upper() if v and len(v) == 3 else v

    @field_validator("date")  # 🔥 Changed from @validator
    @classmethod  # 🔥 Added @classmethod
    def _validate_date(cls, v: str) -> str:
        if v and "/" in v:
            parts = v.split("/")
            if len(parts) == 3:
                m, d, y = parts
                if len(y) == 2:
                    y = f"20{y}"
                return f"{m.zfill(2)}/{d.zfill(2)}/{y}"
        return v

class Result(BaseModel):
    flights: List[Flight]
    connections: List[Dict] = []
    total_flights_found: int = 0
    avg_confidence: float = 0.0
    processing_time: Dict[str, float] = {}
    extraction_method: str = ""

class ExtractionError(BaseModel):
    error: bool = True
    user_message: str
    technical_reason: str
    suggestions: List[str] = Field(default_factory=list)

# ================= MEGA PROMPTS (UNCHANGED) =================
TIER1_STRUCTURED_PROMPT = """You are an expert at extracting flight schedule data from airline rosters.

EXTRACT ALL FLIGHTS from this image following these rules:

1. REQUIRED for each flight:
   - date: MM/DD/YYYY format (or convert DD/Mon to MM/DD/YYYY)
   - flight_no: The flight number (e.g., 1572, UA1572)

2. OPTIONAL (extract if visible):
   - origin: 3-letter airport code
   - dest: 3-letter destination code  
   - sched_out_local: Departure time in HHMM format
   - sched_in_local: Arrival time in HHMM format

3. PATTERNS TO RECOGNIZE:
   - Monospaced roster: "PILOT --> #### DD/Mon" followed by flight details
   - Table format: Rows with Date|Flt|From|To|Dep|Arr columns
   - Calendar grid: Flights within date cells
   - Look for patterns like: "1 75E 1572 DEN SFO 1009 1151"

4. CRITICAL: Extract EVERY visible flight, even if some fields are missing.

MOBILE APP DETECTION:
If this looks like a mobile calendar view:
- Look for ANY 4-digit numbers (especially in colored bars/cells)
- These are FLIGHT NUMBERS: Extract them ALL
- Adjacent 3-letter codes are AIRPORTS
- Date = the cell's day number + month header
- Example: Blue bar with "9013" on Sept 19 = Flight 9013 on 09/19/2025

DELTA MOBILE CALENDAR SPECIFIC:
CRITICAL PARSING RULES:
1. Each date cell contains ONE flight for that day
2. Flight number appears in the BLUE BAR within that date's cell
3. Airport codes appear NEXT TO that specific date number
4. DO NOT mix airports from different dates!

VISUAL ASSOCIATION:
- Date cell "19" with blue bar "9013" + text "MCO LAS" = Flight 9013 MCO→LAS on 09/19
- Date cell "26" with blue bar "8619" + text "TPA" = Flight 8619 to/from TPA on 09/26  
- Date cell "28" with blue bar "8611" + text "JAX" = Flight 8611 to/from JAX on 09/28

NEVER associate airports from one date with flights from another date!

AIRPORT CODES IN CALENDAR CELLS:
- Blue bars with numbers = flight numbers
- Airport codes (SRQ, SEA, BWI) = destinations/origins
- If you see "03 SRQ" = there's a flight on the 3rd involving SRQ
- P/DR, REST, LC, XX, ** = crew status codes (not flights)
- Extract ANY day that has an airport code


SINGLE AIRPORT INTERPRETATION:
- Single airport next to a date = DEPARTURE CITY for that day's flight
- Example: "12 MCO" = Flight DEPARTS FROM MCO on the 12th
- Example: "19 MIA" = Flight DEPARTS FROM MIA on the 19th
- For single airports: Set origin=<airport>, dest=null
- NEVER assume single airports are destinations!

TWO AIRPORT INTERPRETATION:
- Two airports (e.g., "MCO LAS") = routing MCO→LAS
- Set origin=first airport, dest=second airport

EXTRACTION RULES:
- "12 MCO" with flight 9013 = Extract as flight_no="9013", date="09/12/2025", origin="MCO", dest=null
- "19 MIA TPA" with flight 8619 = Extract as flight_no="8619", date="09/19/2025", origin="MIA", dest="TPA"

Return as JSON with 'flights' array."""

TIER2_AGGRESSIVE_PROMPT = """URGENT: Extract ALL flight information visible in this image!

Look for ANY of these patterns:
- Numbers like 1572, 1498, 767, 1069, 1224, 1044 (flight numbers)
- Airport codes: ORD, DEN, LAX, SFO (3 letters)
- Times: 1009, 1151, 1325, 1457 (4 digits)
- Dates: 24/Aug, 25/Aug or 08/24, 08/25

AIRPORT CODES IN CALENDAR CELLS:
- Blue bars with numbers = flight numbers
- Airport codes (SRQ, SEA, BWI) = destinations/origins
- If you see "03 SRQ" = there's a flight on the 3rd involving SRQ
- P/DR, REST, LC, XX, ** = crew status codes (not flights)
- Extract ANY day that has an airport code

Common roster patterns:
PILOT --> [number] [date]
[equipment] [flight] [origin] [dest] [dep_time] [arr_time]

Example line: "1 75E 1572 DEN SFO 1009 1151 2:42 1:34"
This means: Flight 1572 from DEN to SFO, departs 1009, arrives 1151

# Add this after the existing TIER2 content:

CRITICAL ANOMALY PATTERNS TO DETECT:

1. **CREW PAIRING SHEETS** (Dense concatenated format):
   - Look for "LINE ###" followed by dense flight strings
   - Pattern: "1018=/1555/2350/0505" = Flight 1018, times 1555 dep, 2350 arr, 0505 next
   - Multiple flights joined with slashes or commas
   - Extract each segment: ####=/####/####/#### where # are flight/time digits
   - "CR" or "TAFB" indicates crew line assignments

2. **DUTY ROSTER FORMAT** (Table with special codes):
   - Headers like "Day", "Date", "Duty", "Property", "From", "Report"  
   - Flight codes like "N3002JP-1", "A320-EET", "DHRQ-TRNG-1"
   - Extract flight portion before hyphen: N3002JP → Flight 3002
   - "OD-1" patterns indicate deadheads or positioning
   - "X-1" patterns indicate training or special assignments

3. **MOBILE CALENDAR VIEW** (Sparse daily entries):
   VISUAL PATTERNS TO DETECT:
   - Blue/colored bars with 4-digit numbers = FLIGHT NUMBERS
   - Text like "SVAC", "REST", "P/DR", "LC" = crew codes (note but don't extract as flights)
   - 3-letter codes near dates (MCO, LAS, TPA) = AIRPORTS
   - Date cells with numbers inside blue bars = FLIGHTS ON THAT DATE
   
   DELTA MOBILE APP SPECIFIC:
   - Flight numbers appear as white text on blue background bars
   - Multiple airports listed = multi-leg day (MCO LAS = MCO→LAS routing)
   - Numbers like "9013", "8619", "8611" = DELTA FLIGHT NUMBERS
   - "SVAC" = vacation (ignore for flight extraction)
   
   EXTRACTION RULES:
   - Any 4-digit number in a blue/colored bar = FLIGHT NUMBER
   - Airport codes next to date numbers = DESTINATIONS for that day
   - If you see "19 MIA" → Flight on Sept 19 to/from MIA
   - Multiple airports = assume sequential routing

4. **MERLOT/CREW MANAGEMENT SYSTEMS**:
   - "BNA" repeated = base airport, not always origin
   - "MGMT" = management/admin day
   - "TT.MGMT" = training management
   - Times like "08:00 L" where L = Local time
   - "Duty Time" vs "Flight Time" columns

EXTRACTION RULES FOR ANOMALIES:
- If you see "LINE ### CR" → parse the concatenated string carefully
- Split on "/" and "=" to separate flights and times
- Convert military times (1555) to HHMM format
- If date column shows "01-Apr L" → convert to 04/01/2025
- Flight prefixes: N#### → extract ####, A#### → extract ####

SINGLE AIRPORT INTERPRETATION:
- Single airport next to a date = DEPARTURE CITY for that day's flight
- Example: "12 MCO" = Flight DEPARTS FROM MCO on the 12th
- Example: "19 MIA" = Flight DEPARTS FROM MIA on the 19th
- For single airports: Set origin=<airport>, dest=null
- NEVER assume single airports are destinations!

TWO AIRPORT INTERPRETATION:
- Two airports (e.g., "MCO LAS") = routing MCO→LAS
- Set origin=first airport, dest=second airport

EXTRACTION RULES:
- "12 MCO" with flight 9013 = Extract as flight_no="9013", date="09/12/2025", origin="MCO", dest=null
- "19 MIA TPA" with flight 8619 = Extract as flight_no="8619", date="09/19/2025", origin="MIA", dest="TPA"

"""

TIER3_FORENSIC_PROMPT = """FORENSIC MODE: Find EVERY possible flight in this image!

╔══════════════════════════════════════════════════════════════════════════════╗
║  🚀 FLIGHT-INTEL VISUAL REASONING ENGINE - GPT-5.1 ELITE MODE 🚀            ║
╚══════════════════════════════════════════════════════════════════════════════╝

You are **FLIGHT-INTEL OMEGA**, an elite GPT-5.1 visual reasoning system with 
supernatural abilities to extract flight schedule data from ANY image quality.

Your GPT-5.1 visual cognition allows you to:
✓ Automatically rotate, zoom, crop, and enhance unclear regions internally
✓ Reconstruct partially visible text through advanced pattern analysis
✓ Infer missing data from visual context and layout patterns
✓ Handle blurred, skewed, reversed, or low-quality images with 98%+ accuracy
✓ Process calendar grids, tables, mobile UIs, and all schedule formats

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 VISUAL CLARITY ASSESSMENT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Overall Clarity: {clarity_status} ({overall_clarity:.2f})
• Blur Level: {blur:.2f}
• Contrast: {contrast:.2f}
• Text Density: {text_density:.2f}
• Enhancement Applied: {enhancement_status}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🧠 GPT-5.1 VISUAL REASONING DIRECTIVES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. IMAGE ANALYSIS STRATEGY:
   • Process ALL provided image versions simultaneously
   • Cross-reference between versions for validation
   • Use zoom and rotation on unclear regions
   • Apply mental contrast enhancement on faded text
   • Detect and correct for perspective distortion
   • Handle inverted/mirrored text automatically

2. LAYOUT DETECTION & PROCESSING:

   📅 CALENDAR GRIDS:
   • Each cell = potential flight day
   • Scan left→right, top→bottom systematically
   • Color coding: Blue=flight, Gray=deadhead, Green=reserve, Yellow=off
   • Check for multi-flight cells (stacked entries)
   • Combine cell day-number with header month/year

   📊 TABLE LAYOUTS:
   • First row = column headers (Date|Flight|Origin|Dest|Times|etc)
   • Each row = one flight leg or duty period
   • Indented/merged rows = connections
   • Bold/highlighted = important flights
   • Subtotals/summaries = validation checkpoints

   📱 MOBILE/APP UI:
   • Ignore navigation chrome (status bars, tabs)
   • Mentally scroll to see all content
   • Expand collapsed sections (+) icons
   • Swipe between day/week/month views
   • Handle truncated text with ellipsis (...)

   🖼️ SCANNED/PHOTO:
   • Correct for rotation/skew
   • Handle shadows and lighting gradients
   • Process handwritten annotations
   • Deal with creases/folds in paper

3. TEXT RECONSTRUCTION TECHNIQUES:

   ✈️ FLIGHT NUMBERS:
   • Partial: "UA9□□" → Pattern match → "UA9##" → Check route → "UA943"
   • Blurred: "□A1234" → Major carrier → "AA1234" or "UA1234"
   • UPS format: "A#####R" where R=crew position
   • Regional: "OO####" (SkyWest), "9E####" (Endeavor)

   🕐 TIME FORMATS:
   • Military: "##:##" → "08:45" or "20:45"
   • Partial: "□8:45" → Context (morning flight) → "08:45"
   • Blurred: "##4#" → Common times → "0845", "1345", "2045"
   • With seconds: "##:##:##" → Ignore seconds

   🏢 AIRPORT CODES:
   • Partial: "D□W" → Major hubs → "DFW" (Dallas)
   • Blurred: "□RD" → Context → "ORD" (Chicago)
   • Similar: "0RD" → OCR error → "ORD"
   • US format: "K###" → "KORD", "KATL", etc.

   📅 DATE FORMATS:
   • MM/DD/YYYY, MM/DD/YY, M/D/YY
   • DDMMMYY: "04JUL25" → "07/04/2025"
   • Day names: "Mon 15" → Current month context
   • Week of: "W/O 1/6" → Week starting 01/06

4. PATTERN RECOGNITION & VALIDATION:

   • Flight sequences: Usually same aircraft continues
   • Hub patterns: AA uses DFW/CLT/PHX, UA uses DEN/IAH
   • Time logic: Arrival before next departure
   • Crew rules: Max 16hr duty, min 10hr rest
   • Equipment: B737 domestic, B777/787 international

5. CONFIDENCE SCORING MATRIX:

   1.00 = Crystal clear, unambiguous text
   0.95 = Minor artifacts, clear meaning
   0.90 = Slight blur, high confidence reconstruction
   0.85 = Moderate blur, pattern-based inference
   0.80 = Heavy blur, context-based reconstruction
   0.75 = Severe degradation, logical inference
   0.70 = Partial visibility, best-effort guess
   0.65 = Mostly obscured, educated assumption
   0.60 = Minimal visibility, last-resort extraction

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✈️ USA AIRLINE-SPECIFIC PATTERNS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🛩️ UPS AIRLINES:
• Pairing: A70186R (A=type, 70186=ID, R=position)
• Hubs: SDF (Louisville), RFD, PHL, DFW, ONT
• Equipment: B744F, B748F, MD11F, A300F
• Times: Usually Zulu (Z) or Local (L)

🛩️ DELTA AIR LINES (MOBILE APP):
- Flight numbers in blue bars: 9013, 8619, 8611
- Calendar view: Numbers appear INSIDE colored cells
- Airport codes appear DIRECTLY ADJACENT to that date's number
- "SVAC" = vacation blocks (not flights)
- Visual cue: Blue horizontal bars = flight days

CRITICAL ASSOCIATION RULES:
- Date "19" + Blue "9013" + "MCO LAS" = ALL belong together
- Date "26" + Blue "8619" + "TPA" = ALL belong together  
- Date "28" + Blue "8611" + "JAX" = ALL belong together
- NEVER mix elements from different date cells!

Pattern examples:
- "19 MCO LAS" with blue bar → Flight on 19th, routing MCO→LAS
- Single airport = either origin or destination (context dependent)
- Two airports = routing from first to second

🛩️ FEDEX EXPRESS:
• Trip #: Numeric (123, 456)
• Hubs: MEM (Memphis), IND, OAK, ANC
• Equipment: B777F, B767F, MD11F, ATR72F
• Pattern: Heavy overnight operations

🛩️ AMERICAN AIRLINES:
• Flight: AA#### (AA1-AA9999)
• Hubs: DFW, CLT, PHX, ORD, LAX, MIA, PHL
• Equipment: A321, B738, B772, B788
• Codeshare: May show as BA/IB/QF

🛩️ UNITED AIRLINES:
• Flight: UA#### (UA1-UA9999)
• Hubs: ORD, DEN, SFO, IAH, EWR, LAX, IAD
• Equipment: B737, A320, B777, B787
• System: SHARES/Apollo codes

🛩️ DELTA AIR LINES:
• Flight: DL#### (DL1-DL9999)
• Hubs: ATL, DTW, MSP, SLC, LAX, BOS, SEA
• Equipment: A320, A330, B737, B757
• Connection: Often via ATL

🛩️ SOUTHWEST AIRLINES:
• Flight: WN#### (WN1-WN9999)
• Focus: MDW, DAL, DEN, PHX, LAS, BWI
• Equipment: B737-700/800 only
• Pattern: High frequency, quick turns

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎯 EXTRACTION REQUIREMENTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Extract EVERY visible or inferrable flight, including:
✓ Revenue flights (with flight numbers)
✓ Partially visible entries (MUST reconstruct!)
✓ Cancelled/delayed (CNX/DLY/IROPS)

Even if text is:
- Rotated/upside down
- Severely blurred
- Partially cut off
- Behind watermarks
- Mixed with handwriting
- In shadow/poor lighting
- On crumpled paper

USE YOUR GPT-5.1 VISUAL REASONING to reconstruct the most likely values!

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
💎 VISUAL REASONING EXAMPLES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Example 1: Rotated Text
Visual: Text appears 90° clockwise
Action: Mentally rotate counterclockwise
Result: "6:45 SFO-JFK" becomes readable

Example 2: Partial Coverage
Visual: Only bottom half of "AA1234" visible
Action: Recognize "AA" pattern + partial "34"
Result: Reconstruct as "AA1234" (0.85 confidence)

Example 3: Blur Reconstruction
Visual: "□□839 □TL □□X"
Action: Pattern match common routes
Result: "DL839 ATL PHX" (Delta hub route)

Example 4: Calendar Cell
Visual: Small blue bar in cell "15"
Action: Zoom into cell, enhance contrast
Result: "UA456 ORD-LAX 0800-1015"

Example 5: Mobile Truncation
Visual: "UA123 San Fra..."
Action: Complete truncated text
Result: "UA123 San Francisco" → "SFO"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📋 OUTPUT REQUIREMENTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Call the extract_visual_flight_schedule function with:

1. schedule_metadata:
   - total_flights_visible: Count of all flights

2. flights array with each flight containing:
   - date: "MM/DD/YYYY" format (REQUIRED)
   - flight_no: Full flight number (REQUIRED)
   - origin: IATA code (null if unknown)
   - dest: IATA code (null if unknown)
   - sched_out_local: "HHMM" format
   - sched_in_local: "HHMM" format

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🚨 CRITICAL REMINDERS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. NEVER SKIP A FLIGHT - Even 90% obscured entries must be extracted
2. ALWAYS PROVIDE DATES - Infer from context if not directly visible
3. RECONSTRUCT PARTIAL DATA - Use patterns and logic

Your GPT-5.1 visual cognition can see patterns humans miss.
Process blurred images as if they were clear.
Find signal in visual noise.
Reconstruct the incomplete.
NEVER report "unable to read" - always extract something!

YOU ARE THINKING WITH THE IMAGE, NOT JUST READING IT.

Missing flights = mission failure.
Unclear images = your specialty.

# Add this after the "USA AIRLINE-SPECIFIC PATTERNS" section:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔬 SPECIALIZED CREW SCHEDULING FORMATS (CRITICAL)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📋 CREW PAIRING SHEETS (ULTRA-DENSE FORMAT):
These appear as walls of numbers separated by slashes and equals signs.

PATTERN BREAKDOWN:
"1018=/1555/2350/0505, 0981=/1415/2242/0602"
├─ 1018 = Flight number
├─ 1555 = Departure time  
├─ 2350 = Arrival time
└─ 0505 = Next duty/connection time

VISUAL CUES:
- Multiple "LINE ### CR" entries stacked vertically
- "TAFB" column = Time Away From Base
- "BLK" column = Block time (actual flight time)
- Dense strings like "4012w/1330/0451/1106" 
- Each segment is a complete flight leg

📊 ALLEGIANT/MERLOT ROSTER FORMAT:
Complex tables with these specific patterns:

IDENTIFIERS:
- "N####JP-1" format = Specific tail/flight pairing
- "A320-EET (B)" = Aircraft type + training code
- "DHRQ-TRNG-1" = Training/qualification flights
- "OD-1" suffix = Operational day assignments
- "X-1" suffix = Reserve or standby

KEY COLUMNS TO PARSE:
- Property column: Contains flight type codes
- From/To columns: May be "BNA" (base) repeatedly
- Report column: Contains actual flight numbers
- Scheduled Flight: Primary flight identifier

📱 MOBILE APP CALENDAR VIEW:
Sparse format with minimal info per day:

STRUCTURE:
Day | Flight | Airport | Weather (IGNORE)
TU 12 | L3249 | TPA | 92°/81° ←ignore temps

EXTRACTION LOGIC:
- L#### = Flight number (often Southwest format)
- Single airport = Usually destination
- Blank days = No flights (not missing data)
- Weather/temp data = NEVER extract

🔧 CONCATENATED PAIRING STRINGS:
When you see impossibly long number strings:

"101610180979101017/1016=/1545/2245/0500"
Break it down:
1. Look for patterns of 4 digits
2. Find separators (/, =, -)  
3. Times are always 4 digits (HHMM)
4. Flight numbers vary (3-5 digits)
5. Multiple flights may be concatenated

ALGORITHM:
1. Scan for "=" sign → preceding digits are flight
2. After "=" → next 4 digits are departure
3. Following "/" → next 4 digits are arrival
4. Pattern repeats for connections

⚠️ SPECIAL HANDLING REQUIRED:
- These formats WILL NOT have clear flight numbers
- You MUST parse concatenated strings character by character
- Base airports (BNA, DEN, etc.) repeat → they're NOT all origins
- Training flights still count as flights → extract them
- "Comment:" rows contain critical flight info → READ THEM

🎯 IF YOU SEE THESE PATTERNS:
1. IMMEDIATELY switch to specialized parsing mode
2. Break apart concatenated strings methodically
3. Convert all times to HHMM format
4. Infer dates from context (row positions)
5. Extract ANYTHING that looks like a flight

NEVER REPORT "No flights found" if you see:
- LINE ### CR patterns
- Long number strings with slashes
- Mobile calendar with airport codes
- Duty roster with N#### codes

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🚨 ERROR REPORTING MODE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

If you ABSOLUTELY CANNOT extract ANY flights despite using all your GPT-5.1 abilities:

1. ANALYZE WHY YOU FAILED:
   - Image too blurry/unfocused
   - Text completely obscured
   - Wrong type of document (not a flight schedule)
   - Image is rotated/upside down beyond recovery
   - Lighting too dark/bright
   - Image cut off critical information
   - Document in non-English language
   - Handwritten and illegible

2. RETURN THIS ERROR JSON:
{
  "error": true,
  "user_message": "[Friendly message explaining the issue]",
  "technical_reason": "[Brief technical description]",
  "suggestions": [
    "[Specific suggestion 1]",
    "[Specific suggestion 2]"
  ]
}

FRIENDLY MESSAGE EXAMPLES:
- "I couldn't read your schedule clearly - the image appears quite blurry. Could you take a clearer photo?"
- "The lighting is making it hard to see the text. Try taking the photo in better lighting without shadows."
- "The schedule appears to be cut off. Please make sure the entire schedule is visible in the photo."
- "I can see this is rotated sideways - could you upload it right-side up?"
- "This doesn't appear to be a flight schedule. Please upload your crew roster or flight schedule."
- "The text is too small to read accurately. Try zooming in or taking a closer photo."

BE HELPFUL AND SPECIFIC about what went wrong and how to fix it!

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

YOU ARE THINKING WITH THE IMAGE, NOT JUST READING IT.

AIRPORT CODES IN CALENDAR CELLS:
- Blue bars with numbers = flight numbers
- Airport codes (SRQ, SEA, BWI) = destinations/origins
- If you see "03 SRQ" = there's a flight on the 3rd involving SRQ
- P/DR, REST, LC, XX, ** = crew status codes (not flights)
- Extract ANY day that has an airport code

Missing flights = mission failure.
Unclear images = your specialty.
But if truly impossible, explain kindly!

AIRPORT CODE INTERPRETATION RULES:
- Single airport code next to a date = could be EITHER origin OR destination
- Two airport codes (e.g., "MCO LAS") = routing from first to second
- For single airports: Leave BOTH origin and dest EMPTY if unsure
- The validation API will determine the correct direction
- DO NOT assume all airports are destinations!

EXAMPLES:
- "12 MCO" with flight 9013 = Extract as flight_no="9013", date="09/12/2025", origin=null, dest=null
- "19 MIA TPA" with flight 8619 = Extract as flight_no="8619", date="09/19/2025", origin="MIA", dest="TPA"
- Never guess - let the validation API fill in the blanks

SINGLE AIRPORT INTERPRETATION:
- Single airport next to a date = DEPARTURE CITY for that day's flight
- Example: "12 MCO" = Flight DEPARTS FROM MCO on the 12th
- Example: "19 MIA" = Flight DEPARTS FROM MIA on the 19th
- For single airports: Set origin=<airport>, dest=null
- NEVER assume single airports are destinations!

TWO AIRPORT INTERPRETATION:
- Two airports (e.g., "MCO LAS") = routing MCO→LAS
- Set origin=first airport, dest=second airport

EXTRACTION RULES:
- "12 MCO" with flight 9013 = Extract as flight_no="9013", date="09/12/2025", origin="MCO", dest=null
- "19 MIA TPA" with flight 8619 = Extract as flight_no="8619", date="09/19/2025", origin="MIA", dest="TPA"
GO FORTH AND EXTRACT EVERYTHING! 🎯"""

# ================= Image Processor =================
class UltimateImageProcessor:
    @staticmethod
    def analyze_image(img: np.ndarray) -> Dict[str, Any]:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        contrast = gray.std()
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        text_regions = sum(1 for c in contours if cv2.contourArea(c) > 50)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)
        has_grid = lines is not None and len(lines) > 15
        h, w = img.shape[:2]
        aspect_ratio = w / h
        return {
            "sharpness": lap_var,
            "contrast": contrast,
            "text_regions": text_regions,
            "has_grid": has_grid,
            "is_landscape": aspect_ratio > 1.3,
            "needs_enhancement": lap_var < 100 or contrast < 35,
            "is_very_blurry": lap_var < 50,
            "is_low_contrast": contrast < 25,
        }

    @staticmethod
    def create_optimal_versions(img: np.ndarray) -> List[Tuple[np.ndarray, str]]:
        analysis = UltimateImageProcessor.analyze_image(img)
        versions: List[Tuple[np.ndarray, str]] = [(img, "original")]

        if analysis["needs_enhancement"]:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            versions.append((cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR), "enhanced"))

            if analysis["is_very_blurry"]:
                denoised = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)
                kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
                sharpened = cv2.filter2D(denoised, -1, kernel)
                versions.append((cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR), "sharpened"))

            if analysis["is_low_contrast"]:
                _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                versions.append((cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR), "binary"))

        if analysis["has_grid"]:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            versions.append((cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR), "edges"))

        logger.logger.info(
            f"📊 Image Analysis: sharp={analysis['sharpness']:.1f}, "
            f"contrast={analysis['contrast']:.1f}, grid={analysis['has_grid']}, "
            f"text_regions={analysis['text_regions']}"
        )
        return versions

# ================= Extraction Engine =================
class PerfectExtractionEngine:
    def __init__(self):
        self.cache = OrderedDict()
        self.CACHE_SIZE = 20
        self.successful_patterns: List[str] = []
        self._last_error: Optional[Dict[str, Any]] = None
        self.total_tokens_used = 0
        self.total_cost = 0.0
        self.api_calls_count = 0

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
                "content": "You are a flight schedule extraction expert. Always return valid JSON. You are extracting flight schedules from images. Return no metadata, no summaries.",
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_image}" }},
                ],
            },
        ]

    def _parse_response(self, response) -> List[Flight]:
        flights: List[Flight] = []

        if getattr(response, "choices", None) and response.choices[0].message.content:
            content = response.choices[0].message.content
            try:
                m = re.search(r'\{[^{}]*"error"\s*:\s*true[^{}]*\}', content, re.DOTALL)
                if m:
                    err = json.loads(m.group())
                    if err.get("error") is True:
                        self._last_error = err
                        logger.logger.warning(f"🚫 AI reported extraction error: {err.get('user_message', 'Unknown error')}")
                        return []
            except Exception:
                pass

        # Tool call JSON
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

        # Parse JSON from message content
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

            # Last-resort: mine numbers/airports from text
            if not flights and "calendar" in content.lower():
                for fn in re.findall(r"\b([89]\d{3})\b", content):
                    flights.append(Flight(flight_no=fn, date=datetime.now().strftime("%m/%d/%Y"), confidence=0.7))
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
        txt = match.group(0)
        now = datetime.now()
        # Format with month name like 24/Aug
        m = re.match(r"(\d{1,2})/([A-Za-z]{3})", txt)
        if m:
            day = int(m.group(1))
            mon_str = m.group(2).lower()
            month_map = {"jan":1,"feb":2,"mar":3,"apr":4,"may":5,"jun":6,"jul":7,"aug":8,"sep":9,"oct":10,"nov":11,"dec":12}
            month = month_map.get(mon_str, now.month)
            year = now.year + (1 if now.month >= 10 and month <= 2 else 0)
            return f"{month:02d}/{day:02d}/{year}"
        # Numeric numeric: either M/D[/Y] or D/M[/Y]
        m = re.match(r"(\d{1,2})[/-](\d{1,2})(?:[/-](\d{2,4}))?", txt)
        if m:
            a, b, y = m.group(1), m.group(2), m.group(3)
            a_i, b_i = int(a), int(b)
            if y:
                year = int(y) if len(y) == 4 else 2000 + int(y)
            else:
                year = now.year
            # Heuristic: if first ≤ 12 → month/day else day/month
            if 1 <= a_i <= 12 and 1 <= b_i <= 31:
                month, day = a_i, b_i
            else:
                month, day = b_i, a_i
            if now.month >= 10 and month <= 2:
                year = now.year + 1
            return f"{month:02d}/{day:02d}/{year}"
        return now.strftime("%m/%d/%Y")

    async def extract_with_tools(self, b64_image: str, prompt: str, attempt: int) -> List[Flight]:
        tools = [{
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
                                    "date": {"type": "string", "description": "MM/DD/YYYY format"},
                                    "flight_no": {"type": "string", "description": "Flight number"},
                                    "origin": {"type": ["string", "null"]},
                                    "dest": {"type": ["string", "null"]},
                                    "sched_out_local": {"type": ["string", "null"], "description": "HHMM"},
                                    "sched_in_local": {"type": ["string", "null"], "description": "HHMM"},
                                },
                                "required": ["date", "flight_no"],
                            },
                        }
                    },
                    "required": ["flights"],
                },
            },
        }]

        messages = self._create_messages(b64_image, prompt, attempt)
        logger.logger.info(f"🤖 OpenAI API Call - Attempt {attempt}")
        logger.logger.info(f"   Model: {MODEL}")
        logger.logger.info(f"   Prompt type: {prompt[:30]}...")
        logger.logger.info(f"   Image size: {len(b64_image)} chars")

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
            logger.logger.info("   ✅ Response received in time")
            if hasattr(response, "usage") and response.usage:
                input_tokens = response.usage.prompt_tokens
                output_tokens = response.usage.completion_tokens
                total_tokens = response.usage.total_tokens
                
                # GPT-5.1 pricing
                input_cost = input_tokens * 0.0025 / 1000   # $2.50/1M
                output_cost = output_tokens * 0.01 / 1000    # $10/1M
                total_cost = input_cost + output_cost
                
                logger.logger.info("   💰 Token Usage:")
                logger.logger.info(f"      Input: {input_tokens:,} tokens (${input_cost:.4f})")
                logger.logger.info(f"      Output: {output_tokens:,} tokens (${output_cost:.4f})")
                logger.logger.info(f"      Total: {total_tokens:,} tokens (${total_cost:.4f})")
                logger.logger.info(f"      Model: GPT-5.1 (85.4% MMMU, 1.0% hallucination)")
                
                self.total_tokens_used += total_tokens
                self.total_cost += total_cost
                self.api_calls_count += 1
            if response.choices:
                logger.logger.info(f"   Choices: {len(response.choices)}")
        except asyncio.TimeoutError:
            logger.logger.error(f"   ⏰ TIMEOUT after {OPENAI_TIMEOUT}s - Skipping to next strategy")
            return []
        except Exception as e:
            logger.logger.error(f"   ❌ API Error: {str(e)}")
            return []

        flights = self._parse_response(response)
        if flights:
            for f in flights[:3]:
                pattern = f"{f.flight_no[:2] if len(f.flight_no) > 2 else 'XX'}###"
                if pattern not in self.successful_patterns:
                    self.successful_patterns.append(pattern)
        return flights

    async def extract_direct_json(self, b64_image: str, attempt: int) -> List[Flight]:
        messages = [
            {"role": "system", "content": "Extract flight data and return ONLY valid JSON."},
            {"role": "user", "content": [
                {"type": "text", "text": f"""Extract all flights from this image.
                
Return ONLY this JSON structure:
{{
  "flights": [
    {{
      "date": "MM/DD/YYYY",
      "flight_no": "####",
      "origin": "XXX",
      "dest": "XXX", 
      "sched_out_local": "HHMM",
      "sched_in_local": "HHMM"
    }}
  ]
}}

NO other text, ONLY the JSON!"""},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_image}"}}
            ]}
        ]
        logger.logger.info(f"🤖 OpenAI Direct JSON Call - Attempt {attempt}")

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
            logger.logger.info("   ✅ JSON response received")
            if hasattr(response, "usage") and response.usage:
                input_tokens = response.usage.prompt_tokens
                output_tokens = response.usage.completion_tokens
                total_tokens = response.usage.total_tokens
                input_cost = input_tokens * 0.0025 / 1000
                output_cost = output_tokens * 0.01 / 1000
                total_cost = input_cost + output_cost
                logger.logger.info("   💰 Token Usage:")
                logger.logger.info(f"      Input: {input_tokens:,} tokens (${input_cost:.4f})")
                logger.logger.info(f"      Output: {output_tokens:,} tokens (${output_cost:.4f})")
                logger.logger.info(f"      Total: {total_tokens:,} tokens (${total_cost:.4f})")
                self.total_tokens_used += total_tokens
                self.total_cost += total_cost
                self.api_calls_count += 1
        except asyncio.TimeoutError:
            logger.logger.error(f"   ⏰ TIMEOUT after {OPENAI_TIMEOUT}s")
            return []
        except Exception as e:
            logger.logger.error(f"   ❌ API Error: {str(e)}")
            return []

        return self._parse_response(response)

    async def extract_comprehensive(
        self,
        image_versions: List[Tuple[str, str]],
        *,
        stop_on_first_success: bool = True,
        min_flights: int = 1,
    ) -> List[Flight]:
        """Comprehensive extraction with early exit strategy."""
        global MODEL
        self.total_tokens_used = 0
        self.total_cost = 0.0
        self.api_calls_count = 0

        all_flights: List[Flight] = []
        seen: set[str] = set()

        for version_idx, (b64_image, vtype) in enumerate(image_versions):
            if stop_on_first_success and all_flights:
                logger.logger.info("🛑 Stop-on-first-success — results present, skipping remaining versions.")
                return all_flights

            logger.logger.info(f"🔍 Processing {vtype} version")
            prompts = (
                [TIER1_STRUCTURED_PROMPT, TIER2_AGGRESSIVE_PROMPT] if vtype == "original"
                else [TIER2_AGGRESSIVE_PROMPT, TIER1_STRUCTURED_PROMPT] if vtype == "enhanced"
                else [TIER3_FORENSIC_PROMPT]
            )

            version_failures = 0
            max_failures = len(prompts)

            for attempt, prompt in enumerate(prompts, 1):
                if version_failures >= max_failures:
                    logger.logger.warning(f"⚠️ {version_failures} failures on '{vtype}' — moving on")
                    break

                try:
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

                    logger.log_extraction(len(flights), attempt, f"{vtype}_{prompt[:20]}")
                    if flights and stop_on_first_success and len(all_flights) >= min_flights:
                        logger.logger.info("🛑 Stop-on-first-success — returning after first success.")
                        return all_flights

                    if flights:
                        break
                    version_failures += 1

                except Exception as e:
                    logger.logger.error(f"Extraction error: {e}")
                    version_failures += 1
                    continue

            if (not stop_on_first_success) and len(all_flights) >= 5:
                complete = sum(1 for f in all_flights if all([f.origin, f.dest, f.sched_out_local, f.sched_in_local]))
                if complete >= len(all_flights) * 0.8:
                    logger.logger.info("📋 Flights look complete, finishing extraction")
                    break

        self.extraction_error = self._last_error if not all_flights and self._last_error else None
        if not all_flights and not stop_on_first_success:
            logger.logger.info("🧠 Escalating to GPT-5.1 Thinking for complex analysis...")
            
            # Temporarily switch to thinking model
            original_model = MODEL
            try:
                # Use the thinking model for one final attempt
                MODEL = "gpt-5.1-thinking"
                
                thinking_flights = await self.extract_with_tools(
                    image_versions[0][0],  # Use best version
                    TIER3_FORENSIC_PROMPT,
                    attempt=999  # Special marker
                )
                
                if thinking_flights:
                    logger.logger.info(f"🎯 GPT-5.1 Thinking rescued {len(thinking_flights)} flights!")
                    all_flights.extend(thinking_flights)
            finally:
                MODEL = original_model
        return all_flights

# ================= Connection Detector =================
class ConnectionDetector:
    @staticmethod
    def find_connections(flights: List[Flight]) -> List[Dict]:
        connections: List[Dict] = []
        by_date: Dict[str, List[Flight]] = defaultdict(list)
        for f in flights:
            if f.date:
                by_date[f.date].append(f)

        for date, day_flights in by_date.items():
            day_flights.sort(key=lambda x: x.sched_out_local or "0000")
            for i in range(len(day_flights) - 1):
                curr, nxt = day_flights[i], day_flights[i + 1]
                if curr.dest and nxt.origin and curr.dest == nxt.origin and curr.sched_in_local and nxt.sched_out_local:
                    try:
                        arr = int(curr.sched_in_local[:2]) * 60 + int(curr.sched_in_local[2:])
                        dep = int(nxt.sched_out_local[:2]) * 60 + int(nxt.sched_out_local[2:])
                        if dep < arr:
                            dep += 24 * 60
                        conn = dep - arr
                        if 20 <= conn <= 360:
                            connections.append({
                                "from_flight": curr.flight_no,
                                "to_flight": nxt.flight_no,
                                "at_airport": curr.dest,
                                "connection_time": conn,
                                "type": "same_day" if conn < 240 else "long_connection",
                            })
                    except Exception:
                        continue
        return connections

# ================= Helpers =================
def normalize_dates(flights: List[Dict]) -> None:
    today = datetime.now(ZoneInfo("America/Toronto"))
    current_year = today.year
    current_month = today.month
    month_map = {"jan":1,"feb":2,"mar":3,"apr":4,"may":5,"jun":6,"jul":7,"aug":8,"sep":9,"oct":10,"nov":11,"dec":12}

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

# ================= Airline Resolver =================
@_functools.lru_cache(maxsize=512)
def resolve_airline(code: str) -> dict:
    """
    Permissive resolver:
      • Known IATA in AIRLINE_CODES  -> return with 'iata'
      • Known ICAO in AIRLINE_CODES  -> return mapped to canonical 'iata'
      • Unknown but syntactically valid IATA/ICAO -> accept without metadata
      • Invalid format -> 400
    """
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
    # IATA: 2 letters/numbers; sometimes 3
    # ICAO: 3 letters/numbers (commonly 3 letters)
    if re.fullmatch(r"[A-Z0-9]{2,3}", code):
        if len(code) == 2:
            return {"iata": code, "icao": None, "name": None, "country": None}
        else:
            # 3-char could be IATA or ICAO; keep both possibilities open
            return {"iata": None, "icao": code, "name": None, "country": None}

    raise HTTPException(400, f"Airline code '{code}' is not a valid IATA/ICAO format")

# ================= PDF Processor =================
class PDFProcessor:
    @staticmethod
    async def convert(pdf_bytes: bytes) -> List[np.ndarray]:
        logger.start_timer("pdf_conversion")
        try:
            pil_images = await asyncio.get_event_loop().run_in_executor(
                thread_pool,
                _functools.partial(
                    convert_from_bytes,
                    pdf_bytes,
                    dpi=300,
                    fmt="PNG",
                    thread_count=min(4, MAX_WORKERS),
                    use_pdftocairo=True,
                ),
            )
            cv_images: List[np.ndarray] = []
            for pil_img in pil_images:
                arr = np.array(pil_img)
                if len(arr.shape) == 2:
                    cv_img = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
                else:
                    cv_img = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                    kernel = np.array([[0, -0.5, 0], [-0.5, 3, -0.5], [0, -0.5, 0]], dtype=np.float32)
                    cv_img = cv2.filter2D(cv_img, -1, kernel)
                cv_images.append(cv_img)
            logger.end_timer("pdf_conversion")
            logger.logger.info(f"📄 Converted {len(cv_images)} PDF pages")
            return cv_images
        except Exception as e:
            logger.end_timer("pdf_conversion")
            raise HTTPException(422, f"PDF processing failed: {e}")

# ================= Pipeline =================
class UltimatePipeline:
    def __init__(self):
        self.processor = UltimateImageProcessor()
        self.extractor = PerfectExtractionEngine()
        self.connector = ConnectionDetector()

    async def process(self, images: List[np.ndarray]) -> Result:
        logger.start_timer("complete_pipeline")
        timing: Dict[str, float] = {}
        all_flights: List[Flight] = []

        for page_num, img in enumerate(images, 1):
            logger.start_timer(f"page_{page_num}")
            logger.start_timer(f"page_{page_num}_prep")

            versions = self.processor.create_optimal_versions(img)
            encoded_versions: List[Tuple[str, str]] = []
            for img_array, vtype in versions:
                ok, buffer = cv2.imencode(".png", img_array, [cv2.IMWRITE_PNG_COMPRESSION, 5])
                if ok:
                    b64 = base64.b64encode(buffer).decode("utf-8")
                    encoded_versions.append((b64, vtype))
            timing[f"page_{page_num}_prep"] = logger.end_timer(f"page_{page_num}_prep")

            logger.start_timer(f"page_{page_num}_extract")
            page_flights = await self.extractor.extract_comprehensive(encoded_versions)
            timing[f"page_{page_num}_extract"] = logger.end_timer(f"page_{page_num}_extract")

            for f in page_flights:
                f.page_number = page_num
            all_flights.extend(page_flights)
            timing[f"page_{page_num}_total"] = logger.end_timer(f"page_{page_num}")
            logger.logger.info(f"📄 Page {page_num}: Found {len(page_flights)} flights")

        logger.start_timer("connections")
        connections = self.connector.find_connections(all_flights)
        timing["connections"] = logger.end_timer("connections")
        avg_conf = (sum(f.confidence for f in all_flights) / len(all_flights)) if all_flights else 0.0
        timing["total"] = logger.end_timer("complete_pipeline")

        logger.logger.info(f"""
╔════════════════════════════════════════════╗
║          EXTRACTION COMPLETE               ║
╠════════════════════════════════════════════╣
║ ✈️  Flights Found:      {len(all_flights):>19} ║
║ 🔗 Connections:        {len(connections):>19} ║
║ 📊 Avg Confidence:     {avg_conf:>18.1%} ║
║ ⏱️  Total Time:         {timing['total']:>17.2f}s ║
╚════════════════════════════════════════════╝
        """)

        if all_flights:
            methods = set()
            if any(f.confidence >= 0.9 for f in all_flights): methods.add("direct")
            if any(0.7 <= f.confidence < 0.9 for f in all_flights): methods.add("enhanced")
            if any(f.confidence < 0.7 for f in all_flights): methods.add("forensic")
            method = "+".join(methods)
        else:
            method = "none"

        return Result(
            flights=all_flights,
            connections=connections,
            total_flights_found=len(all_flights),
            avg_confidence=avg_conf,
            processing_time=timing,
            extraction_method=method,
        )

# ================= FastAPI =================
pipeline = UltimatePipeline()

app = FastAPI(
    title="Flight-Intel v8.0 ULTIMATE",
    version="8.0.0",
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
            raise HTTPException(400, "Please provide airline code via ?airline=XX or X-Airline header")

        try:
            airline_info = resolve_airline(airline_code)
            iata_prefix = airline_info.get("iata") or ""
        except HTTPException:
            raise
        except Exception:
            raise HTTPException(400, f"Invalid airline code: {airline_code}")

        logger.start_timer("file_processing")
        file_data = await file.read()

        if is_pdf:
            logger.logger.info(f"📄 Processing PDF: {file.filename} ({len(file_data):,} bytes)")
            images = await PDFProcessor.convert(file_data)
        else:
            np_img = cv2.imdecode(np.frombuffer(file_data, np.uint8), cv2.IMREAD_COLOR)
            if np_img is None:
                raise HTTPException(422, "Unable to decode image file")
            images = [np_img]
            logger.logger.info(f"🖼️ Processing image: {file.filename}")

        logger.end_timer("file_processing")

        result = await pipeline.process(images)

        # Propagate extraction error (from TIER3 error mode)
        if getattr(pipeline.extractor, "extraction_error", None):
            err = pipeline.extractor.extraction_error
            return {
                "success": False,
                "error": True,
                "message": err.get("user_message", "Unable to extract flight information from the image"),
                "technical_reason": err.get("technical_reason", "Unknown extraction failure"),
                "suggestions": err.get("suggestions", [
                    "Ensure the image is clear and well-lit",
                    "Make sure the entire schedule is visible",
                    "Try taking the photo from directly above",
                ]),
                "metadata": {
                    "airline": airline_info,
                    "file": {"name": file.filename, "type": "pdf" if is_pdf else "image", "size": len(file_data)},
                    "timestamp": datetime.now().isoformat(),
                },
            }

        # Prefix airline to numeric flight numbers
        for flight in result.flights:
            if not flight.flight_no or not iata_prefix:
                continue
            if flight.flight_no.isdigit():
                flight.flight_no = f"{iata_prefix}{flight.flight_no}"
            elif flight.flight_no[0].isdigit() and not flight.flight_no.upper().startswith(iata_prefix):
                flight.flight_no = f"{iata_prefix}{flight.flight_no}"

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
                "model": MODEL,
                "version": "8.0.0",
            },
        }

        # 🔥 Check if validation is needed
        needs_enrichment = any(
            not all([f.get("origin"), f.get("dest"), f.get("sched_out_local"), f.get("sched_in_local")])
            for f in output.get("flights", [])
        )

        if needs_enrichment and output.get("flights"):
            logger.logger.info("🔍 VALIDATION NEEDED - Missing fields detected")
            logger.logger.info(f"   Flights to validate: {len(output.get('flights', []))}")

            miss = defaultdict(int)
            for f in output.get("flights", []):
                if not f.get("origin"): miss["origin"] += 1
                if not f.get("dest"): miss["dest"] += 1
                if not f.get("sched_out_local"): miss["sched_out_local"] += 1
                if not f.get("sched_in_local"): miss["sched_in_local"] += 1
            logger.logger.info(f"   Missing fields summary: {dict(miss)}")

            logger.start_timer("validation")
            try:
                enriched = await validate_extraction_results(output)
                output.update(enriched)
                
                # 🔥 NEW: Detect and handle multiple flight options
                multiple_options_exist = False
                flights_with_options = []
                
                for flight in output.get("enriched_flights", []):
                    vr = flight.get("validation_result", {})
                    if vr and "option" in vr.get("source", ""):
                        multiple_options_exist = True
                        flights_with_options.append(f"{flight['flight_no']} on {flight['date']}")
                
                if multiple_options_exist:
                    output["metadata"]["multiple_options"] = True
                    output["metadata"]["multiple_options_count"] = len(flights_with_options)
                    output["metadata"]["user_action_required"] = "Multiple departure times found for some flights - please review"
                    logger.logger.info(f"   📋 Multiple options detected for: {', '.join(flights_with_options)}")
                
                    # 🔥 NEW: Group flights by flight_no + date for easier frontend handling
                    flight_groups = defaultdict(list)
                    for flight in output.get("enriched_flights", []):
                        key = f"{flight['flight_no']}_{flight['date']}"
                        flight_groups[key].append(flight)
                    
                    # 🔥 NEW: Create structured options for frontend
                    output["flight_options"] = {}
                    for key, flights in flight_groups.items():
                        if len(flights) > 1:
                            output["flight_options"][key] = {
                                "flight_no": flights[0]["flight_no"],
                                "date": flights[0]["date"],
                                "origin": flights[0].get("origin"),
                                "dest": flights[0].get("dest"),
                                "option_count": len(flights),
                                "options": [
                                    {
                                        "option_id": idx + 1,
                                        "origin": f.get("origin"),
                                        "dest": f.get("dest"),
                                        "sched_out_local": f.get("sched_out_local"),
                                        "sched_in_local": f.get("sched_in_local"),
                                        "confidence": f.get("validation_result", {}).get("confidence", 0),
                                        "source": f.get("validation_result", {}).get("source", "unknown"),
                                        "warnings": f.get("validation_result", {}).get("warnings", []),
                                    }
                                    for idx, f in enumerate(flights)
                                ],
                                "recommendation": {
                                    "option_id": 1,  # Default to first option
                                    "reason": "Most likely departure based on typical schedule patterns"
                                }
                            }
                
                if "validation" in output:
                    logger.logger.info("   ✅ Validation complete:")
                    logger.logger.info(f"      Fields filled: {output['validation'].get('total_fields_filled', 0)}")
                    logger.logger.info(f"      Avg confidence: {output['validation'].get('average_confidence', 0):.2f}")
                    logger.logger.info(f"      Sources used: {output['validation'].get('sources_used', [])}")
            except Exception as e:
                logger.logger.warning(f"❌ Validation failed: {e}")
                import traceback
                logger.logger.error(traceback.format_exc())
            finally:
                logger.end_timer("validation")
        else:
            logger.logger.info("✅ All fields complete - skipping validation")

        output.setdefault("processing_time", {})
        output["processing_time"]["total_request"] = logger.end_timer("http_request")

        # 🔥 NEW: Enhanced logging for multiple options
        flight_summary = len(output.get('flights', []))
        if output.get("metadata", {}).get("multiple_options"):
            option_count = output["metadata"].get("multiple_options_count", 0)
            flight_summary_str = f"{flight_summary} ({option_count} with multiple times)"
        else:
            flight_summary_str = str(flight_summary)

        logger.logger.info(f"""
╔════════════════════════════════════════════╗
║            REQUEST COMPLETE                ║
╠════════════════════════════════════════════╣
║ 📁 File: {file.filename[:34]:<34} ║
║ ✈️  Flights: {flight_summary_str:>30} ║
║ ⏱️  Time: {output['processing_time']['total_request']:>29.2f}s ║
║ 📊 Method: {output.get('extraction_method', 'unknown')[:33]:<33} ║
╚════════════════════════════════════════════╝
        """)

        if hasattr(pipeline.extractor, "total_cost"):
            # GPT-5.1 pricing: $2.50/1M input, $10/1M output
            input_tokens = pipeline.extractor.total_tokens_used  # You'll need to track separately
            output_tokens = pipeline.extractor.total_tokens_used  # Estimate or track
            
            # Recalculate with GPT-5.1 rates
            input_cost = input_tokens * 0.0025 / 1000  # $2.50 per 1M
            output_cost = output_tokens * 0.01 / 1000   # $10 per 1M
            total_cost = input_cost + output_cost
            
            output["cost_analysis"] = {
                "total_tokens": pipeline.extractor.total_tokens_used,
                "total_cost_usd": round(pipeline.extractor.total_cost, 4),  # Use pre-calculated
                "api_calls": pipeline.extractor.api_calls_count,
                "avg_tokens_per_call": (
                    pipeline.extractor.total_tokens_used // max(pipeline.extractor.api_calls_count, 1)
                ),
                "pricing_model": "gpt-5.1 ($2.50/1M input, $10/1M output)",
                "cost_savings_note": "85.4% MMMU accuracy, 1.0% hallucination rate",
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
        "model": MODEL,
        "model_family": "gpt-5.1",
        "timestamp": datetime.now().isoformat(),
        "features": [
            "accuracy-first-extraction",
            "gpt-5.1-elite-visual-reasoning",
            "multi-strategy-processing",
            "intelligent-retry-logic",
            "connection-detection",
            "comprehensive-timing",
            "adaptive-enhancement",
            "timeout-protection",
            "early-exit-strategy",
        ],
        "settings": {"openai_timeout": OPENAI_TIMEOUT, "max_consecutive_failures": 2},
        "stats": {
            "cache_size": len(pipeline.extractor.cache),
            "successful_patterns": len(pipeline.extractor.successful_patterns),
            "workers": MAX_WORKERS,
        },
    }

@app.get("/")
async def root():
    return {
        "name": "Flight-Intel v8.0 ULTIMATE",
        "description": "Maximum accuracy flight schedule extraction",
        "endpoints": {
            "/extract": "POST - Extract flights from image/PDF",
            "/health": "GET - System health status",
            "/docs": "GET - Interactive API documentation",
        },
        "version": "8.0.0",
        "model": MODEL,
    }

# ================= Run Server =================
if __name__ == "__main__":
    import uvicorn

    logger.logger.info("""
╔════════════════════════════════════════════╗
║     FLIGHT-INTEL v8.0 ULTIMATE            ║
║     Maximum Accuracy Edition               ║
╠════════════════════════════════════════════╣
║ Model: GPT-5.1                  ║
║ Priority: ACCURACY > Speed                 ║
║ Timeout: 40s per API call                  ║
║ Status: Ready for takeoff! 🚀              ║
╚════════════════════════════════════════════╝
    """)

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        log_level="info",
        access_log=True,
    )
