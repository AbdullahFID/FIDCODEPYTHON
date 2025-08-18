# main.py
# Flight-Intel v8.0 ULTIMATE — Maximum Accuracy with Speed Optimization
# Fixed: Added timeouts, API logging, and early exit strategy

from dotenv import load_dotenv
load_dotenv()

import os, base64, cv2, numpy as np, re, json, asyncio, functools, concurrent
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Any, Pattern
from collections import defaultdict, OrderedDict
import time
import hashlib

from fastapi import FastAPI, UploadFile, File, HTTPException, Body, Query, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field, validator

from openai import AsyncOpenAI
from pdf2image import convert_from_bytes
from zoneinfo import ZoneInfo
import logging
import functools as _functools

# External helpers
from flight_intel_patch import (
    validate_extraction_results
)
from airlines import AIRLINE_CODES

# ================= Enhanced Logging with Timing =================
class PerfectLogger:
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.timers = {}
        self.stats = defaultdict(list)
    
    def start_timer(self, operation: str) -> float:
        """Start a timer for an operation"""
        start_time = time.perf_counter()
        self.timers[operation] = start_time
        self.logger.info(f"⏱️  [{operation}] Started")
        return start_time
    
    def end_timer(self, operation: str) -> float:
        """End a timer and return elapsed time"""
        if operation not in self.timers:
            return 0.0
        
        elapsed = time.perf_counter() - self.timers[operation]
        self.stats[operation].append(elapsed)
        
        # Use emoji based on speed
        emoji = "⚡" if elapsed < 1 else "✅" if elapsed < 3 else "⏰"
        self.logger.info(f"{emoji} [{operation}] Completed in {elapsed:.2f}s")
        
        del self.timers[operation]
        return elapsed
    
    def log_extraction(self, flights_count: int, attempt: int, method: str):
        """Log extraction results"""
        if flights_count > 0:
            self.logger.info(f"✈️  Extracted {flights_count} flights on attempt {attempt} using {method}")
        else:
            self.logger.warning(f"⚠️  No flights found on attempt {attempt} using {method}")
    
    def log_api_call(self, api_name: str, params: Dict, response: Any = None):
        """Log API calls with params and response"""
        self.logger.info(f"🌐 API CALL: {api_name}")
        self.logger.info(f"   📤 Params: {json.dumps(params, default=str)[:500]}")
        if response:
            self.logger.info(f"   📥 Response: {json.dumps(response, default=str)[:500]}")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = PerfectLogger("flight-intel")

# ================= Configuration =================
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_KEY:
    raise RuntimeError("❌ OPENAI_API_KEY missing")

MODEL = "o4-mini-2025-04-16"
MAX_TOKENS = 3000  # Optimal for o4-mini
OPENAI_TIMEOUT = 40  # 40 second timeout for OpenAI calls

# Performance settings
MAX_WORKERS = min(32, (os.cpu_count() or 4) * 2)
thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS)

client = AsyncOpenAI(api_key=OPENAI_KEY)

# ================= Regex Patterns (Pre-compiled for speed) =================
class Patterns:
    """Pre-compiled regex patterns for maximum speed"""
    FLIGHT_NO = re.compile(r'\b(?:[A-Z]{1,2}\d{1,5}[A-Z]?|[A-Z]\d{5}[A-Z]?|\d{3,5})\b')
    DATE_DMY = re.compile(r'\b(\d{1,2})[/-](\d{1,2})(?:[/-](\d{2,4}))?\b')
    DATE_MDY = re.compile(r'\b(\d{1,2})/([A-Za-z]{3})\b')
    TIME_24H = re.compile(r'\b([01]?\d|2[0-3]):?([0-5]\d)\b')
    AIRPORT = re.compile(r'\b[A-Z]{3}\b')
    ROUTE = re.compile(r'\b([A-Z]{3})\s*[-–→>]\s*([A-Z]{3})\b')
    
patterns = Patterns()

# ================= Data Models =================
class Flight(BaseModel):
    date: str = Field(..., description="MM/DD/YYYY format")
    flight_no: str = Field(..., description="Flight number")
    origin: Optional[str] = None
    dest: Optional[str] = None
    sched_out_local: Optional[str] = None  # HHMM format
    sched_in_local: Optional[str] = None   # HHMM format
    page_number: Optional[int] = None
    confidence: float = Field(default=1.0, ge=0, le=1)

    @validator("flight_no")
    def clean_flight_no(cls, v):
        if not v:
            return v
        # Clean and standardize
        cleaned = re.sub(r'[^\w\d]', '', v.upper())
        return cleaned

    @validator("origin", "dest")
    def validate_airport(cls, v):
        if v and len(v) == 3:
            return v.upper()
        return v

    @validator("date")
    def validate_date(cls, v):
        # Ensure MM/DD/YYYY format
        if v and '/' in v:
            parts = v.split('/')
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

# ================= PERFECTED PROMPTS =================
# Three-tier prompt strategy for maximum accuracy

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

Return as JSON with 'flights' array."""

TIER2_AGGRESSIVE_PROMPT = """URGENT: Extract ALL flight information visible in this image!

Look for ANY of these patterns:
- Numbers like 1572, 1498, 767, 1069, 1224, 1044 (flight numbers)
- Airport codes: ORD, DEN, LAX, SFO (3 letters)
- Times: 1009, 1151, 1325, 1457 (4 digits)
- Dates: 24/Aug, 25/Aug or 08/24, 08/25

Common roster patterns:
PILOT --> [number] [date]
[equipment] [flight] [origin] [dest] [dep_time] [arr_time]

Example line: "1 75E 1572 DEN SFO 1009 1151 2:42 1:34"
This means: Flight 1572 from DEN to SFO, departs 1009, arrives 1151

EXTRACT EVERYTHING! Return JSON with 'flights' array."""

TIER3_FORENSIC_PROMPT = """FORENSIC MODE: Find EVERY possible flight in this image!

╔══════════════════════════════════════════════════════════════════════════════╗
║  🚀 FLIGHT-INTEL VISUAL REASONING ENGINE - O4-MINI HIGH MODE 🚀              ║
╚══════════════════════════════════════════════════════════════════════════════╝

You are **FLIGHT-INTEL OMEGA**, an elite O4-Mini visual reasoning system with 
supernatural abilities to extract flight schedule data from ANY image quality.

Your O4-Mini visual cognition allows you to:
✓ Automatically rotate, zoom, crop, and enhance unclear regions internally
✓ Reconstruct partially visible text through advanced pattern analysis
✓ Infer missing data from visual context and layout patterns
✓ Handle blurred, skewed, reversed, or low-quality images with 95%+ accuracy
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
🧠 O4-MINI VISUAL REASONING DIRECTIVES
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
   • Hub patterns: AA uses DFW/CLT/PHX, UA uses ORD/DEN/IAH
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

USE YOUR O4-MINI VISUAL REASONING to reconstruct the most likely values!

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

Your O4-Mini visual cognition can see patterns humans miss.
Process blurred images as if they were clear.
Find signal in visual noise.
Reconstruct the incomplete.
NEVER report "unable to read" - always extract something!

YOU ARE THINKING WITH THE IMAGE, NOT JUST READING IT.

Missing flights = mission failure.
Unclear images = your specialty.
GO FORTH AND EXTRACT EVERYTHING! 🎯"""

# ================= Ultimate Image Processor =================
class UltimateImageProcessor:
    """Advanced image processing with multiple enhancement strategies"""
    
    @staticmethod
    def analyze_image(img: np.ndarray) -> Dict[str, Any]:
        """Deep image analysis for optimal processing strategy"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Calculate metrics
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        contrast = gray.std()
        
        # Text detection
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        text_regions = len([c for c in contours if cv2.contourArea(c) > 50])
        
        # Grid detection
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
        has_grid = lines is not None and len(lines) > 15
        
        # Determine if it's likely a roster
        height, width = img.shape[:2]
        aspect_ratio = width / height
        is_landscape = aspect_ratio > 1.3
        
        return {
            "sharpness": laplacian_var,
            "contrast": contrast,
            "text_regions": text_regions,
            "has_grid": has_grid,
            "is_landscape": is_landscape,
            "needs_enhancement": laplacian_var < 100 or contrast < 35,
            "is_very_blurry": laplacian_var < 50,
            "is_low_contrast": contrast < 25
        }
    
    @staticmethod
    def create_optimal_versions(img: np.ndarray) -> List[Tuple[np.ndarray, str]]:
        """Create multiple optimized versions for different extraction strategies"""
        analysis = UltimateImageProcessor.analyze_image(img)
        versions = []
        
        # Always include original
        versions.append((img, "original"))
        
        # Adaptive enhancement based on analysis
        if analysis["needs_enhancement"]:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Enhanced contrast version
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            versions.append((cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR), "enhanced"))
            
            # Super enhancement for very poor images
            if analysis["is_very_blurry"]:
                # Denoise + sharpen
                denoised = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)
                kernel = np.array([[-1,-1,-1],
                                   [-1, 9,-1],
                                   [-1,-1,-1]])
                sharpened = cv2.filter2D(denoised, -1, kernel)
                versions.append((cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR), "sharpened"))
            
            # Binary version for text extraction
            if analysis["is_low_contrast"]:
                _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                versions.append((cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR), "binary"))
        
        # Grid-specific processing
        if analysis["has_grid"]:
            # Perspective correction might help
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            versions.append((cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR), "edges"))
        
        logger.logger.info(f"📊 Image Analysis: sharp={analysis['sharpness']:.1f}, "
                          f"contrast={analysis['contrast']:.1f}, grid={analysis['has_grid']}, "
                          f"text_regions={analysis['text_regions']}")
        
        return versions

# ================= PERFECT EXTRACTION ENGINE =================
class PerfectExtractionEngine:
    """Multi-strategy extraction engine for maximum accuracy"""
    
    def __init__(self):
        self.cache = OrderedDict()
        self.CACHE_SIZE = 20  # Smaller cache for freshness
        self.successful_patterns = []  # Learn from successes
    
    def _create_messages(self, b64_image: str, prompt: str, attempt: int) -> List[dict]:
        """Create optimized messages for o4-mini"""
        
        # Add successful pattern hints if we have them
        if self.successful_patterns and attempt > 1:
            prompt += f"\n\nPreviously successful patterns: {', '.join(self.successful_patterns[:3])}"
        
        return [
            {
                "role": "system", 
                "content": "You are a flight schedule extraction expert. Always return valid JSON. You are extracting flight schedules from images. Return no metadata, no summaries."
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_image}"}}
                ]
            }
        ]
    
    def _parse_response(self, response) -> List[Flight]:
        """Robust response parsing with multiple strategies"""
        flights = []
        
        # Strategy 1: Parse tool calls
        for choice in response.choices:
            if hasattr(choice.message, 'tool_calls') and choice.message.tool_calls:
                for tc in choice.message.tool_calls:
                    if tc.function and tc.function.arguments:
                        try:
                            data = json.loads(tc.function.arguments)
                            for f in data.get("flights", []):
                                flight = Flight(**f)
                                flights.append(flight)
                        except Exception as e:
                            logger.logger.debug(f"Tool parse error: {e}")
        
        # Strategy 2: Parse JSON from content
        if not flights and response.choices:
            content = response.choices[0].message.content
            if content:
                # Try to find JSON in the response
                json_patterns = [
                    r'\{[\s\S]*\}',  # Standard JSON
                    r'```json\s*([\s\S]*?)\s*```',  # Markdown JSON
                    r'```\s*([\s\S]*?)\s*```',  # Generic code block
                ]
                
                for pattern in json_patterns:
                    matches = re.findall(pattern, content)
                    for match in matches:
                        try:
                            # Clean the match
                            if isinstance(match, tuple):
                                match = match[0]
                            cleaned = match.strip()
                            if cleaned.startswith('```'):
                                cleaned = cleaned[3:]
                            if cleaned.endswith('```'):
                                cleaned = cleaned[:-3]
                            if cleaned.startswith('json'):
                                cleaned = cleaned[4:]
                            
                            data = json.loads(cleaned)
                            if isinstance(data, dict) and "flights" in data:
                                for f in data["flights"]:
                                    flight = Flight(**f)
                                    flights.append(flight)
                                    break
                        except Exception as e:
                            continue
                
                # Strategy 3: Extract structured data from text
                if not flights:
                    flights.extend(self._extract_from_text(content))
        
        return flights
    
    def _extract_from_text(self, text: str) -> List[Flight]:
        """Extract flights from unstructured text response"""
        flights = []
        
        # Look for flight patterns in text
        lines = text.split('\n')
        for line in lines:
            # Pattern: flight_no origin dest date time
            flight_match = patterns.FLIGHT_NO.search(line)
            if flight_match:
                flight_data = {"flight_no": flight_match.group()}
                
                # Find airports
                airports = patterns.AIRPORT.findall(line)
                if len(airports) >= 2:
                    flight_data["origin"] = airports[0]
                    flight_data["dest"] = airports[1]
                
                # Find date
                date_match = patterns.DATE_DMY.search(line) or patterns.DATE_MDY.search(line)
                if date_match:
                    # Convert to MM/DD/YYYY
                    flight_data["date"] = self._parse_date(date_match)
                
                # Find times
                times = patterns.TIME_24H.findall(line)
                if times:
                    if len(times) >= 1:
                        flight_data["sched_out_local"] = times[0][0] + times[0][1]
                    if len(times) >= 2:
                        flight_data["sched_in_local"] = times[1][0] + times[1][1]
                
                # Only add if we have minimum required fields
                if "date" in flight_data or len(flights) < 5:  # Be aggressive if we have few flights
                    if "date" not in flight_data:
                        flight_data["date"] = datetime.now().strftime("%m/%d/%Y")
                    
                    try:
                        flight = Flight(**flight_data)
                        flights.append(flight)
                    except:
                        pass
        
        return flights
    
    def _parse_date(self, match) -> str:
        """Convert various date formats to MM/DD/YYYY"""
        groups = match.groups()
        current_year = datetime.now().year
        
        if len(groups) == 3:  # DD/MM/YYYY or MM/DD/YYYY
            d, m, y = groups
            if y:
                year = int(y) if len(y) == 4 else 2000 + int(y)
            else:
                year = current_year
            
            # Assume MM/DD for US format
            return f"{m.zfill(2)}/{d.zfill(2)}/{year}"
        
        return datetime.now().strftime("%m/%d/%Y")
    
    async def extract_with_tools(self, b64_image: str, prompt: str, attempt: int) -> List[Flight]:
        """Extract using function calling with TIMEOUT"""
        
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
                                    "sched_out_local": {"type": ["string", "null"], "description": "HHMM format"},
                                    "sched_in_local": {"type": ["string", "null"], "description": "HHMM format"}
                                },
                                "required": ["date", "flight_no"]
                            }
                        }
                    },
                    "required": ["flights"]
                }
            }
        }]
        
        messages = self._create_messages(b64_image, prompt, attempt)
        
        # Log OpenAI API call
        logger.logger.info(f"🤖 OpenAI API Call - Attempt {attempt}")
        logger.logger.info(f"   Model: {MODEL}")
        logger.logger.info(f"   Prompt type: {prompt[:30]}...")
        logger.logger.info(f"   Image size: {len(b64_image)} chars")
        
        try:
            # Add timeout wrapper
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    tools=tools,
                    tool_choice={"type": "function", "function": {"name": "extract_flights"}},
                    max_completion_tokens=MAX_TOKENS,
                    # temperature=1 is the ONLY supported value for o4-mini
                    n=2 if attempt > 1 else 1  # Get multiple responses on retry
                ),
                timeout=OPENAI_TIMEOUT
            )
            
            # Log response
            logger.logger.info(f"   ✅ Response received in time")
            if response.choices:
                logger.logger.info(f"   Choices: {len(response.choices)}")
            
        except asyncio.TimeoutError:
            logger.logger.error(f"   ⏰ TIMEOUT after {OPENAI_TIMEOUT}s - Skipping to next strategy")
            return []
        except Exception as e:
            logger.logger.error(f"   ❌ API Error: {str(e)}")
            return []
        
        flights = self._parse_response(response)
        
        # Track successful patterns
        if flights:
            for f in flights[:3]:  # Track first few patterns
                pattern = f"{f.flight_no[:2] if len(f.flight_no) > 2 else 'XX'}###"
                if pattern not in self.successful_patterns:
                    self.successful_patterns.append(pattern)
        
        return flights
    
    async def extract_direct_json(self, b64_image: str, attempt: int) -> List[Flight]:
        """Direct JSON extraction with TIMEOUT"""
        
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
                    response_format={"type": "json_object"}  # Force JSON response
                ),
                timeout=OPENAI_TIMEOUT
            )
            logger.logger.info(f"   ✅ JSON response received")
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
        min_flights: int = 1
    ) -> List[Flight]:
        """Comprehensive extraction with early exit strategy (no slow-response bail)."""

        all_flights: List[Flight] = []
        seen_flights: set[str] = set()

        # Try each version with different strategies
        for version_idx, (b64_image, version_type) in enumerate(image_versions):

            # Early short-circuit if configured
            if stop_on_first_success and all_flights:
                logger.logger.info("🛑 Stop-on-first-success — results already present, skipping remaining versions.")
                return all_flights

            logger.logger.info(f"🔍 Processing {version_type} version")

            # Strategy sequence based on version type
            if version_type == "original":
                # Keep TIER3 out of original to avoid over-detection on clean images
                prompts = [TIER1_STRUCTURED_PROMPT, TIER2_AGGRESSIVE_PROMPT]
            elif version_type == "enhanced":
                prompts = [TIER2_AGGRESSIVE_PROMPT, TIER1_STRUCTURED_PROMPT]
            else:
                # e.g., edges/binary/etc. → the forensic sweep
                prompts = [TIER3_FORENSIC_PROMPT]

            # Failures are scoped to this version
            version_failures = 0
            max_failures_per_version = len(prompts)  # try all prompts unless we hit real exceptions

            for attempt, prompt in enumerate(prompts, 1):
                if version_failures >= max_failures_per_version:
                    logger.logger.warning(f"⚠️ {version_failures} failures on '{version_type}' — moving to next version")
                    break

                try:
                    timer = logger.start_timer(f"extract_{version_type}_attempt_{attempt}")
                    flights = await self.extract_with_tools(b64_image, prompt, attempt)
                    elapsed = logger.end_timer(f"extract_{version_type}_attempt_{attempt}")

                    # NOTE: No slow-response bail — we continue regardless of elapsed

                    # Only try the JSON fallback if we still have NOTHING overall
                    if (not flights) and (attempt == len(prompts)) and (not all_flights):
                        timer = logger.start_timer(f"extract_{version_type}_json")
                        flights = await self.extract_direct_json(b64_image, attempt)
                        logger.end_timer(f"extract_{version_type}_json")

                    # Deduplicate and collect
                    for flight in flights:
                        flight_key = f"{flight.flight_no}_{flight.date}"
                        if flight_key not in seen_flights:
                            seen_flights.add(flight_key)
                            flight.confidence = 1.0 - (0.1 * (version_idx + attempt - 1))
                            all_flights.append(flight)

                    logger.log_extraction(len(flights), attempt, f"{version_type}_{prompt[:20]}")

                    if flights:
                        if stop_on_first_success and len(all_flights) >= min_flights:
                            logger.logger.info("🛑 Stop-on-first-success — returning early after first successful extraction.")
                            return all_flights
                        # Success for this version → move to the next version
                        break
                    else:
                        version_failures += 1

                except Exception as e:
                    logger.logger.error(f"Extraction error: {e}")
                    version_failures += 1
                    continue

            # Exhaustive mode only: completeness heuristic
            if (not stop_on_first_success) and len(all_flights) >= 5:
                logger.logger.info(f"✅ Found {len(all_flights)} flights, checking if more needed...")
                complete_flights = sum(
                    1 for f in all_flights if all([f.origin, f.dest, f.sched_out_local, f.sched_in_local])
                )
                if complete_flights >= len(all_flights) * 0.8:
                    logger.logger.info("📋 Flights look complete, finishing extraction")
                    break

        return all_flights



# ================= Connection Detector =================
class ConnectionDetector:
    """Intelligent connection detection"""
    
    @staticmethod
    def find_connections(flights: List[Flight]) -> List[Dict]:
        """Find connections between flights"""
        connections = []
        
        # Group by date
        by_date = defaultdict(list)
        for f in flights:
            if f.date:
                by_date[f.date].append(f)
        
        for date, day_flights in by_date.items():
            # Sort by departure time
            day_flights.sort(key=lambda x: x.sched_out_local or "0000")
            
            for i in range(len(day_flights) - 1):
                curr, next = day_flights[i], day_flights[i+1]
                
                # Check for connection
                if (curr.dest and next.origin and 
                    curr.dest == next.origin and
                    curr.sched_in_local and next.sched_out_local):
                    
                    try:
                        # Calculate connection time
                        arr_h, arr_m = int(curr.sched_in_local[:2]), int(curr.sched_in_local[2:])
                        dep_h, dep_m = int(next.sched_out_local[:2]), int(next.sched_out_local[2:])
                        
                        arr_minutes = arr_h * 60 + arr_m
                        dep_minutes = dep_h * 60 + dep_m
                        
                        # Handle day rollover
                        if dep_minutes < arr_minutes:
                            dep_minutes += 24 * 60
                        
                        connection_time = dep_minutes - arr_minutes
                        
                        if 20 <= connection_time <= 360:  # 20 min to 6 hours
                            connections.append({
                                "from_flight": curr.flight_no,
                                "to_flight": next.flight_no,
                                "at_airport": curr.dest,
                                "connection_time": connection_time,
                                "type": "same_day" if connection_time < 240 else "long_connection"
                            })
                    except:
                        pass
        
        return connections

# ================= Date Normalizer =================
def normalize_dates(flights: List[Dict]) -> None:
    """Normalize dates to MM/DD/YYYY format with correct year"""
    
    today = datetime.now(ZoneInfo("America/Toronto"))
    current_year = today.year
    current_month = today.month
    
    month_map = {
        "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
        "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12
    }
    
    for f in flights:
        date_str = f.get("date", "")
        if not date_str:
            continue
        
        # Handle DD/Mon format (e.g., "24/Aug")
        if '/' in date_str and any(mon in date_str.lower() for mon in month_map):
            parts = date_str.split('/')
            if len(parts) == 2:
                day = parts[0].strip()
                mon_str = parts[1].strip().lower()[:3]
                if mon_str in month_map:
                    month = month_map[mon_str]
                    year = current_year
                    
                    # Handle year rollover
                    if current_month >= 10 and month <= 2:
                        year += 1
                    
                    f["date"] = f"{month:02d}/{day.zfill(2)}/{year}"
                    continue
        
        # Handle MM/DD/YYYY or MM/DD format
        if '/' in date_str:
            parts = date_str.split('/')
            if len(parts) >= 2:
                try:
                    month = int(parts[0])
                    day = int(parts[1])
                    year = int(parts[2]) if len(parts) > 2 else current_year
                    
                    # Fix 2-digit years
                    if year < 100:
                        year += 2000
                    
                    # Handle year rollover
                    if current_month >= 10 and month <= 2:
                        year = current_year + 1
                    elif year < current_year:
                        year = current_year
                    
                    f["date"] = f"{month:02d}/{day:02d}/{year}"
                except:
                    pass

# ================= Airline Resolver =================
@functools.lru_cache(maxsize=512)
def resolve_airline(code: str) -> dict:
    """Resolve airline code to full info"""
    code = code.upper()
    
    # Direct IATA lookup
    if code in AIRLINE_CODES:
        return {"iata": code, **AIRLINE_CODES[code]}
    
    # ICAO lookup
    for iata, data in AIRLINE_CODES.items():
        if data.get("icao") == code:
            return {"iata": iata, **data}
    
    raise HTTPException(404, f"Unknown airline code '{code}'")

# ================= PDF Processor =================
class PDFProcessor:
    """Optimized PDF processing"""
    
    @staticmethod
    async def convert(pdf_bytes: bytes) -> List[np.ndarray]:
        """Convert PDF to images"""
        timer = logger.start_timer("pdf_conversion")
        
        try:
            # Optimal DPI for balance of quality and speed
            pil_images = await asyncio.get_event_loop().run_in_executor(
                thread_pool,
                _functools.partial(
                    convert_from_bytes,
                    pdf_bytes,
                    dpi=250,  # Good balance
                    fmt="PNG",
                    thread_count=min(4, MAX_WORKERS),
                    use_pdftocairo=True
                )
            )
            
            # Convert to OpenCV format
            cv_images = []
            for pil_img in pil_images:
                arr = np.array(pil_img)
                if len(arr.shape) == 2:
                    cv_img = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
                else:
                    cv_img = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                cv_images.append(cv_img)
            
            elapsed = logger.end_timer("pdf_conversion")
            logger.logger.info(f"📄 Converted {len(cv_images)} PDF pages")
            
            return cv_images
            
        except Exception as e:
            logger.end_timer("pdf_conversion")
            raise HTTPException(422, f"PDF processing failed: {e}")

# ================= MAIN PIPELINE =================
class UltimatePipeline:
    """The ultimate extraction pipeline"""
    
    def __init__(self):
        self.processor = UltimateImageProcessor()
        self.extractor = PerfectExtractionEngine()
        self.connector = ConnectionDetector()
    
    async def process(self, images: List[np.ndarray]) -> Result:
        """Process images with maximum accuracy"""
        
        pipeline_timer = logger.start_timer("complete_pipeline")
        timing_info = {}
        
        all_flights = []
        
        # Process each page
        for page_num, img in enumerate(images, 1):
            page_timer = logger.start_timer(f"page_{page_num}")
            
            # Create optimized versions
            prep_timer = logger.start_timer(f"page_{page_num}_prep")
            versions = self.processor.create_optimal_versions(img)
            
            # Encode versions
            encoded_versions = []
            for img_array, version_type in versions:
                ok, buffer = cv2.imencode(".png", img_array, 
                                         [cv2.IMWRITE_PNG_COMPRESSION, 5])
                if ok:
                    b64 = base64.b64encode(buffer).decode("utf-8")
                    encoded_versions.append((b64, version_type))
            
            timing_info[f"page_{page_num}_prep"] = logger.end_timer(f"page_{page_num}_prep")
            
            # Extract flights
            extract_timer = logger.start_timer(f"page_{page_num}_extract")
            page_flights = await self.extractor.extract_comprehensive(encoded_versions)
            timing_info[f"page_{page_num}_extract"] = logger.end_timer(f"page_{page_num}_extract")
            
            # Add page number
            for f in page_flights:
                f.page_number = page_num
            
            all_flights.extend(page_flights)
            timing_info[f"page_{page_num}_total"] = logger.end_timer(f"page_{page_num}")
            
            logger.logger.info(f"📄 Page {page_num}: Found {len(page_flights)} flights")
        
        # Find connections
        conn_timer = logger.start_timer("connections")
        connections = self.connector.find_connections(all_flights)
        timing_info["connections"] = logger.end_timer("connections")
        
        # Calculate average confidence
        avg_confidence = (sum(f.confidence for f in all_flights) / len(all_flights) 
                         if all_flights else 0.0)
        
        timing_info["total"] = logger.end_timer("complete_pipeline")
        
        # Log summary
        logger.logger.info(f"""
╔════════════════════════════════════════════╗
║          EXTRACTION COMPLETE               ║
╠════════════════════════════════════════════╣
║ ✈️  Flights Found:      {len(all_flights):>19} ║
║ 🔗 Connections:        {len(connections):>19} ║
║ 📊 Avg Confidence:     {avg_confidence:>18.1%} ║
║ ⏱️  Total Time:         {timing_info['total']:>17.2f}s ║
╚════════════════════════════════════════════╝
        """)
        
        # Determine extraction method
        if all_flights:
            methods = set()
            if any(f.confidence >= 0.9 for f in all_flights):
                methods.add("direct")
            if any(0.7 <= f.confidence < 0.9 for f in all_flights):
                methods.add("enhanced")
            if any(f.confidence < 0.7 for f in all_flights):
                methods.add("forensic")
            extraction_method = "+".join(methods)
        else:
            extraction_method = "none"
        
        return Result(
            flights=all_flights,
            connections=connections,
            total_flights_found=len(all_flights),
            avg_confidence=avg_confidence,
            processing_time=timing_info,
            extraction_method=extraction_method
        )

# ================= FastAPI Application =================
pipeline = UltimatePipeline()

app = FastAPI(
    title="Flight-Intel v8.0 ULTIMATE",
    version="8.0.0",
    description="Maximum accuracy flight extraction with o4-mini"
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

@app.post("/extract", response_model=None)
async def extract_flights(
    file: UploadFile = File(..., description="Image or PDF file"),
    airline: Optional[str] = Query(None, description="Airline code (IATA/ICAO)"),
    x_airline: Optional[str] = Header(None, alias="X-Airline"),
):
    """Extract flights with maximum accuracy"""
    
    request_timer = logger.start_timer("http_request")
    request_start = time.perf_counter()
    
    try:
        # Validate file type
        content_type = (file.content_type or "").lower()
        is_pdf = content_type == "application/pdf"
        is_image = content_type.startswith("image/")
        
        if not (is_pdf or is_image):
            raise HTTPException(415, "Please upload an image (PNG/JPEG) or PDF file")
        
        # Get airline code
        airline_code = airline or x_airline
        if not airline_code:
            raise HTTPException(400, "Please provide airline code via ?airline=XX or X-Airline header")
        
        # Resolve airline
        try:
            airline_info = resolve_airline(airline_code)
            iata_prefix = airline_info["iata"]
        except:
            raise HTTPException(400, f"Invalid airline code: {airline_code}")
        
        # Read and process file
        file_timer = logger.start_timer("file_processing")
        file_data = await file.read()
        
        images = []
        if is_pdf:
            logger.logger.info(f"📄 Processing PDF: {file.filename} ({len(file_data):,} bytes)")
            images = await PDFProcessor.convert(file_data)
        else:
            # Decode image
            np_img = cv2.imdecode(np.frombuffer(file_data, np.uint8), cv2.IMREAD_COLOR)
            if np_img is None:
                raise HTTPException(422, "Unable to decode image file")
            images = [np_img]
            logger.logger.info(f"🖼️ Processing image: {file.filename}")
        
        logger.end_timer("file_processing")
        
        # Run extraction pipeline
        result = await pipeline.process(images)
        
        # Apply airline prefix
        for flight in result.flights:
            if flight.flight_no:
                # Add prefix if needed
                if flight.flight_no.isdigit():
                    flight.flight_no = f"{iata_prefix}{flight.flight_no}"
                elif not flight.flight_no.startswith(iata_prefix):
                    if flight.flight_no[0].isdigit():
                        flight.flight_no = f"{iata_prefix}{flight.flight_no}"
        
        # Prepare response
        output = jsonable_encoder(result)
        
        # Normalize dates
        normalize_dates(output.get("flights", []))
        
        # Add metadata
        output["metadata"] = {
            "airline": airline_info,
            "file": {
                "name": file.filename,
                "type": "pdf" if is_pdf else "image",
                "size": len(file_data),
                "pages": len(images) if is_pdf else 1
            },
            "processing": {
                "started": datetime.fromtimestamp(request_start).isoformat(),
                "model": MODEL,
                "version": "8.0.0"
            }
        }
        
        # Validation if incomplete
        needs_enrichment = any(
            not all([f.get("origin"), f.get("dest"), 
                    f.get("sched_out_local"), f.get("sched_in_local")])
            for f in output.get("flights", [])
        )
        
        if needs_enrichment and output.get("flights"):
            logger.logger.info("🔍 VALIDATION NEEDED - Missing fields detected")
            logger.logger.info(f"   Flights to validate: {len(output.get('flights', []))}")
            
            val_timer = logger.start_timer("validation")
            try:
                # Log what we're sending to validation
                missing_fields_summary = defaultdict(int)
                for f in output.get("flights", []):
                    if not f.get("origin"):
                        missing_fields_summary["origin"] += 1
                    if not f.get("dest"):
                        missing_fields_summary["dest"] += 1
                    if not f.get("sched_out_local"):
                        missing_fields_summary["sched_out_local"] += 1
                    if not f.get("sched_in_local"):
                        missing_fields_summary["sched_in_local"] += 1
                
                logger.logger.info(f"   Missing fields summary: {dict(missing_fields_summary)}")
                
                # Call validation
                enriched = await validate_extraction_results(output)
                output.update(enriched)
                
                # Log validation results
                if "validation" in output:
                    logger.logger.info(f"   ✅ Validation complete:")
                    logger.logger.info(f"      Fields filled: {output['validation'].get('total_fields_filled', 0)}")
                    logger.logger.info(f"      Avg confidence: {output['validation'].get('average_confidence', 0):.2f}")
                    logger.logger.info(f"      Sources used: {output['validation'].get('sources_used', [])}")
                
            except Exception as e:
                logger.logger.warning(f"❌ Validation failed: {e}")
            finally:
                logger.end_timer("validation")
        else:
            logger.logger.info("✅ All fields complete - skipping validation")
        
        # Final timing
        output["processing_time"]["total_request"] = logger.end_timer("http_request")
        
        # Success summary
        logger.logger.info(f"""
╔════════════════════════════════════════════╗
║            REQUEST COMPLETE                ║
╠════════════════════════════════════════════╣
║ 📁 File: {file.filename[:34]:<34} ║
║ ✈️  Flights: {len(output.get('flights', [])):>30} ║
║ ⏱️  Time: {output['processing_time']['total_request']:>29.2f}s ║
║ 📊 Method: {output.get('extraction_method', 'unknown')[:33]:<33} ║
╚════════════════════════════════════════════╝
        """)
        
        return output
        
    except HTTPException:
        raise
    except Exception as e:
        logger.logger.error(f"Request failed: {e}", exc_info=True)
        raise HTTPException(500, f"Processing failed: {str(e)}")

@app.get("/health")
async def health_check():
    """System health check"""
    return {
        "status": "healthy",
        "version": "8.0.0",
        "model": MODEL,
        "timestamp": datetime.now().isoformat(),
        "features": [
            "accuracy-first-extraction",
            "multi-strategy-processing",
            "intelligent-retry-logic",
            "connection-detection",
            "comprehensive-timing",
            "adaptive-enhancement",
            "timeout-protection",
            "early-exit-strategy"
        ],
        "settings": {
            "openai_timeout": OPENAI_TIMEOUT,
            "max_consecutive_failures": 2
        },
        "stats": {
            "cache_size": len(pipeline.extractor.cache),
            "successful_patterns": len(pipeline.extractor.successful_patterns),
            "workers": MAX_WORKERS
        }
    }

@app.get("/")
async def root():
    """API information"""
    return {
        "name": "Flight-Intel v8.0 ULTIMATE",
        "description": "Maximum accuracy flight schedule extraction",
        "endpoints": {
            "/extract": "POST - Extract flights from image/PDF",
            "/health": "GET - System health status",
            "/docs": "GET - Interactive API documentation"
        },
        "version": "8.0.0",
        "model": MODEL
    }

# ================= Run Server =================
if __name__ == "__main__":
    import uvicorn
    
    logger.logger.info("""
╔════════════════════════════════════════════╗
║     FLIGHT-INTEL v8.0 ULTIMATE            ║
║     Maximum Accuracy Edition               ║
╠════════════════════════════════════════════╣
║ Model: o4-mini-2025-04-16                  ║
║ Priority: ACCURACY > Speed                 ║
║ Timeout: 40s per API call                  ║
║ Status: Ready for takeoff! 🚀              ║
╚════════════════════════════════════════════╝
    """)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        access_log=True
    )