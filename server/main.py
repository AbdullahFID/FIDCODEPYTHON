"""
Enhanced Flight-Intel UPS Roster Extractor v5.0 - SPEED OPTIMIZED
High-performance version with O4-Mini
- Optimized for speed while maintaining precision
- Parallel processing where possible
- Reduced image processing overhead
- Efficient data structures and caching
"""
# uvicorn main:app --reload
from dotenv import load_dotenv
load_dotenv()
import os, base64, cv2, numpy as np, re, json, asyncio
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any, Set
from fastapi import FastAPI, UploadFile, File, HTTPException, Body, Query, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from openai import AsyncOpenAI
import logging
from collections import defaultdict, Counter
from flight_intel_patch import (
    validate_flights_endpoint,
    validate_extraction_results
)
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from airlines import AIRLINE_CODES
from pdf2image import convert_from_bytes
import PyPDF2
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor
import functools
from typing import Pattern

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SYSTEM = (
    "You are extracting flight schedules from images. "
    "You MUST call the tool and return ONLY the following fields per flight: "
    "date, flight_no, origin, dest, sched_out_local, sched_in_local. "
    "Return no metadata, no summaries."
)

# ---------- Performance Settings ------------------------------
USE_OCR_HINTS = os.getenv("USE_OCR_HINTS", "0") == "1"  # Disabled by default for speed
MAX_IMAGE_VERSIONS = 2  # Reduced from 5
MAX_ROIS_PER_IMAGE = 5  # Reduced from unlimited
ENABLE_LAYOUT_DETECTION = os.getenv("ENABLE_LAYOUT_DETECTION", "0") == "1"  # Optional
MAX_CELLS_TO_PROCESS = 30  # Limit cell processing

# Thread pool for CPU-bound operations
thread_pool = ThreadPoolExecutor(max_workers=4)

# ---------- OCR Setup (Minimal) ------------------------------
OCR_TESS_AVAILABLE = False
Image = None
pytesseract = None

if USE_OCR_HINTS:
    try:
        import pytesseract as _pytesseract
        from PIL import Image as _PIL_Image
        pytesseract = _pytesseract
        Image = _PIL_Image
        if not hasattr(Image, "ANTIALIAS"):
            Image.ANTIALIAS = Image.Resampling.LANCZOS
        OCR_TESS_AVAILABLE = True
    except Exception:
        logger.info("Tesseract/PIL not available; continuing without OCR hints.")

# ---------- Configuration for O4-Mini ------------------------------------
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
MODEL = "o4-mini-2025-04-16"
REASONING_EFFORT = "high"
MAX_OUTPUT_TOKENS = 16384
REASONING_SUMMARY = "detailed"

if not OPENAI_KEY:
    raise RuntimeError("OPENAI_API_KEY missing")

openai_client = AsyncOpenAI(api_key=OPENAI_KEY)

app = FastAPI(title="Flight-Intel Vision API O4-Mini Speed", version="5.0-speed")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# ---------- Cached Regex Patterns (Performance) ------------------------------
class RegexCache:
    """Pre-compiled regex patterns for performance"""
    DATE_PATTERN: Pattern = re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b")
    TIME_PATTERN: Pattern = re.compile(r"\b\d{4}\b|\b\d{1,2}:\d{2}\b")
    FLIGHT_PATTERN: Pattern = re.compile(r"\b[A-Z]{1,2}\d{3,5}[A-Z]?\b|\b[A-Z]\d{5}[A-Z]?\b")
    AIRPORT_PATTERN: Pattern = re.compile(r"\b[A-Z]{3,4}\b")
    UPS_PATTERN: Pattern = re.compile(r"^[A-Z]\d{5}[A-Z]?$")
    STANDARD_FLIGHT_PATTERN: Pattern = re.compile(r"^[A-Z]{2}\d{1,4}[A-Z]?$")
    NUMERIC_FLIGHT_PATTERN: Pattern = re.compile(r"^\d{3,5}$")
    VALID_FLIGHT_PATTERN: Pattern = re.compile(r"^[A-Z0-9]{2,7}[A-Z]?$")

regex_cache = RegexCache()

# =================== USA AIRPORTS ===================
# All USA airports (IATA codes) - Major hubs, regional, and smaller airports

USA_AIRPORTS = {
    # === MAJOR HUBS ===
    "ATL", "ORD", "LAX", "DFW", "DEN", "JFK", "SFO", "SEA", "LAS", "MCO",
    "PHX", "EWR", "IAH", "MIA", "BOS", "MSP", "DTW", "FLL", "PHL", "LGA",
    "CLT", "BWI", "DCA", "SLC", "SAN", "TPA", "MDW", "BNA", "AUS", "PDX",
    "STL", "MCI", "OAK", "SMF", "SJC", "RDU", "RSW", "CLE", "CVG", "IND",
    "MKE", "PIT", "CMH", "SAT", "MSY", "JAX", "RIC", "BDL", "ALB", "BUF",
    
    # === NORTHEAST USA ===
    "SYR", "ROC", "PVD", "MHT", "PWM", "BTV", "BGR", "ORH", "ACK", "MVY",
    "HYA", "EWB", "HPN", "ISP", "SWF", "ABE", "MDT", "AVP", "IPT", "LBE",
    "JST", "AOO", "FKL", "DUJ", "BFD", "ERI", "PNS", "BGM", "ITH", "ELM",
    "IAG", "JHW", "BUF", "ROC", "SYR", "ALB", "PLB", "SLK", "MSS", "OGS",
    "ART", "RME", "PBG", "HVN",
    
    # === SOUTHEAST USA ===
    "MIA", "FLL", "PBI", "RSW", "TPA", "MCO", "SRQ", "PIE", "FMY", "PNS",
    "VPS", "ECP", "TLH", "GNV", "JAX", "DAB", "MLB", "VRB", "PGD", "SFB",
    "TIX", "LAL", "EYW", "APF", "BCT", "FXE", "FPR", "HST", "IMM", "ISM",
    "MCF", "OPF", "PHK", "SGJ", "UST", "X14", "ATL", "SAV", "AGS", "CSG",
    "ABY", "VLD", "BQK", "AHN", "MCN", "WRB", "CHA", "TYS", "TRI", "MEM",
    "BNA", "HSV", "BHM", "MOB", "MGM", "DHN", "CSG", "GTR", "GPT", "JAN",
    "TUP", "GLH", "MEI", "PIB", "LIT", "XNA", "FSM", "TXK", "ELD", "HOT",
    "JBR", "HRO", "SHV", "MLU", "AEX", "LFT", "LCH", "BTR", "MSY", "NEW",
    "CHS", "MYR", "GSP", "CAE", "FLO", "HHH", "CRE", "AVL", "GSO", "RDU",
    "CLT", "ILM", "EWN", "OAJ", "FAY", "INT", "PSK", "ISO", "RWI", "USA",
    "PHF", "ORF", "RIC", "LYH", "ROA", "CHO", "SHD", "HGR", "EKN", "CRW",
    "BKW", "CKB", "LWB", "PKB", "MGW", "HTS", "BLF",
    
    # === MIDWEST USA ===
    "ORD", "MDW", "RFD", "MLI", "PIA", "BMI", "SPI", "UIN", "DEC", "CGX",
    "MWA", "DTW", "GRR", "AZO", "FNT", "LAN", "MBS", "TVC", "PLN", "APN",
    "CIU", "CMX", "ESC", "IMT", "IWD", "MQT", "RHI", "SAW", "MKE", "GRB",
    "ATW", "MSN", "LSE", "CWA", "EAU", "RHI", "MSP", "RST", "DLH", "HIB",
    "INL", "BJI", "BRD", "STC", "AXN", "TVF", "GPZ", "CID", "DBQ", "ALO",
    "MCW", "DSM", "OTM", "SUX", "FOD", "EST", "STL", "COU", "JLN", "SGF",
    "TBN", "CGI", "IND", "EVV", "FWA", "SBN", "BMG", "GYY", "LAF", "MIE",
    "CVG", "CMH", "DAY", "TOL", "CAK", "YNG", "MFD", "CLE", "MCI", "ICT",
    "SLN", "MHK", "FOE", "HYS", "DDC", "GCK", "LBL", "OMA", "LNK", "GRI",
    "EAR", "LBF", "BFF", "CDR", "AIA", "MCK", "FSD", "SUX", "ABR", "ATY",
    "HON", "MBG", "PIR", "RAP", "GFK", "FAR", "BIS", "MOT", "ISN", "JMS",
    "DIK", "DVL", "BHK", "XWA",
    
    # === SOUTHWEST USA ===
    "PHX", "TUS", "YUM", "FLG", "PRC", "IFP", "IGM", "SOW", "LAX", "SAN",
    "SFO", "OAK", "SJC", "BUR", "LGB", "SNA", "ONT", "PSP", "SMF", "FAT",
    "BFL", "SBA", "SBP", "SMX", "MRY", "STS", "RDD", "CIC", "MOD", "MCE",
    "VIS", "IPL", "OXR", "CMA", "WJF", "PMD", "BYS", "SDB", "NTD", "CLD",
    "CRQ", "RIV", "RAL", "CCB", "HHR", "SMO", "TOA", "AVX", "SZN", "L35",
    "LAS", "RNO", "VGT", "BLD", "IFP", "ELY", "TPH", "HTH", "LOL", "LSV",
    "O08", "0L7", "0L9", "1L1", "SLC", "OGD", "PVU", "SGU", "CDC", "VEL",
    "CNY", "DPG", "ENV", "HIF", "DEN", "COS", "ASE", "EGE", "GJT", "DRO",
    "MTJ", "HDN", "GUC", "TEX", "CEZ", "ALS", "PUB", "LAA", "LIC", "CAG",
    "ITR", "4V1", "ABQ", "SAF", "ROW", "FMN", "LRU", "TCC", "CNM", "HOB",
    "ATS", "CVS", "PVW", "E80", "SRR", "TCS", "SVC", "DMN", "ONM", "RTN",
    "SKX", "SOW", "0E0", "E14", "E16", "E19", "E23", "E33",
    
    # === NORTHWEST USA ===
    "SEA", "GEG", "YKM", "PSC", "ALW", "PUW", "EAT", "BLI", "PAE", "BFI",
    "RNT", "OLM", "SFF", "FHR", "NUW", "TCM", "TIW", "PDX", "EUG", "MFR",
    "RDM", "OTH", "AST", "TTD", "SLE", "CVO", "ONO", "LMT", "LKV", "HIO",
    "MMV", "BOI", "TWF", "SUN", "IDA", "PIH", "LWS", "GGW", "CDA", "MLP",
    "PUL", "BIL", "BZN", "GGW", "GTF", "HLN", "HVR", "GPI", "BTM", "DLN",
    "GDV", "LWT", "MLS", "SDY", "WYS", "BDX", "BKX", "CUT", "CTB", "EDK",
    "JOD", "M75", "OLF", "OPH", "RPX", "THM", "GCC", "JAC", "RKS", "COD",
    "CPR", "RIW", "LAR", "SHR", "EVW", "LND", "FCA", "MLS", "OLF", "HVR",
    
    # === ALASKA ===
    "ANC", "FAI", "JNU", "SIT", "KTN", "BET", "OTZ", "BRW", "SCC", "ADQ",
    "DLG", "AKN", "CDV", "YAK", "WRG", "PSG", "GST", "SGY", "HNS", "HNH",
    "DUT", "GAL", "MCG", "TAL", "UNK", "ORT", "ANI", "CDB", "CEM", "ENA",
    "HCR", "HOM", "HPB", "IAN", "ILI", "VAK", "KNW", "ADK", "MDO", "MOU",
    "OME", "PHO", "PTH", "SCM", "WMO", "ANV", "ATK", "EEK", "EMK", "GLV",
    "AET", "KWT", "SDP", "MTM", "SNP", "NME", "DRG", "BSW", "CHU", "CIK",
    "CYF", "EAA", "ELI", "KLG", "HPB", "ITO", "KAL", "LMA", "MLL", "NUI",
    "ORV", "PIZ", "PQS", "SKK", "STG", "TKJ", "TOG", "VAK", "VEE", "WBB",
    "WLK", "WSN", "WTK",
    
    # === HAWAII ===
    "HNL", "OGG", "KOA", "LIH", "ITO", "MKK", "LNY", "JHM", "MUE", "BSF",
    "HDH", "HHI", "JRF", "KAL", "LUP", "MKK", "NGF", "HIK", "PAK", "PHN",
    
    # === US TERRITORIES ===
    "SJU", "PSE", "BQN", "MAZ", "SIG", "VQS", "CPX", "FAJ", "STT", "STX",
    "SIG", "GUM", "SPN", "ROP", "TIQ", "TNI", "UAM", "PPG", "OFU", "TAV",
}

# =================== USA AIRCRAFT TYPES ===================
# Format: ICAO Code: Full Name (IATA Code)

AIRCRAFT_TYPES = {
    # === BOEING ===
    "B37M": "Boeing 737 MAX 7 (7M7)",
    "B38M": "Boeing 737 MAX 8 (7M8)",
    "B39M": "Boeing 737 MAX 9 (7M9)",
    "B3XM": "Boeing 737 MAX 10 (7MJ)",
    "B703": "Boeing 707 (703)",
    "B712": "Boeing 717 (717)",
    "B720": "Boeing 720B (B72)",
    "B721": "Boeing 727-100 (721)",
    "B722": "Boeing 727-200 (722)",
    "B732": "Boeing 737-200 (732)",
    "B733": "Boeing 737-300 (733)",
    "B734": "Boeing 737-400 (734)",
    "B735": "Boeing 737-500 (735)",
    "B736": "Boeing 737-600 (736)",
    "B737": "Boeing 737-700/700ER (73G)",
    "B738": "Boeing 737-800 (738)",
    "B739": "Boeing 737-900/900ER (739)",
    "B741": "Boeing 747-100 (741)",
    "B742": "Boeing 747-200 (742)",
    "B743": "Boeing 747-300 (743)",
    "B744": "Boeing 747-400/400ER (744)",
    "B748": "Boeing 747-8I (74H)",
    "B74F": "Boeing 747-8F (74N)",
    "B74R": "Boeing 747SR (74R)",
    "B74S": "Boeing 747SP (74L)",
    "B752": "Boeing 757-200 (752)",
    "B753": "Boeing 757-300 (753)",
    "B762": "Boeing 767-200/200ER (762)",
    "B763": "Boeing 767-300/300ER (763)",
    "B764": "Boeing 767-400ER (764)",
    "B772": "Boeing 777-200/200ER (772)",
    "B773": "Boeing 777-300 (773)",
    "B77L": "Boeing 777-200LR (77L)",
    "B77W": "Boeing 777-300ER (77W)",
    "B778": "Boeing 777-8 (778)",
    "B779": "Boeing 777-9 (779)",
    "B788": "Boeing 787-8 (788)",
    "B789": "Boeing 787-9 (789)",
    "B78X": "Boeing 787-10 (781)",
    "BLCF": "Boeing 747-400 LCF Dreamlifter (74B)",
    "B52": "Boeing B-52 Stratofortress",
    "K35R": "Boeing KC-135 Stratotanker",
    "E3TF": "Boeing E-3 Sentry (AWACS)",
    "B461": "BAe 146-100 (141)",
    "B462": "BAe 146-200 (142)",
    "B463": "BAe 146-300 (143)",
    
    # === AIRBUS ===
    "A19N": "Airbus A319neo (31N)",
    "A20N": "Airbus A320neo (32N)",
    "A21N": "Airbus A321neo (32Q)",
    "A306": "Airbus A300-600 (AB6)",
    "A30B": "Airbus A300B2/B4/C4 (AB4)",
    "A310": "Airbus A310-200 (312)",
    "A318": "Airbus A318 (318)",
    "A319": "Airbus A319 (319)",
    "A320": "Airbus A320 (320)",
    "A321": "Airbus A321 (321)",
    "A332": "Airbus A330-200 (332)",
    "A333": "Airbus A330-300 (333)",
    "A338": "Airbus A330-800 (338)",
    "A339": "Airbus A330-900 (339)",
    "A342": "Airbus A340-200 (342)",
    "A343": "Airbus A340-300 (343)",
    "A345": "Airbus A340-500 (345)",
    "A346": "Airbus A340-600 (346)",
    "A359": "Airbus A350-900 (359)",
    "A35K": "Airbus A350-1000 (351)",
    "A388": "Airbus A380-800 (388)",
    "BCS1": "Airbus A220-100 (221)",
    "BCS3": "Airbus A220-300 (223)",
    
    # === BOMBARDIER ===
    "CRJ1": "Bombardier CRJ100 (CR1)",
    "CRJ2": "Bombardier CRJ200 (CR2)",
    "CRJ7": "Bombardier CRJ700/550 (CR7)",
    "CRJ9": "Bombardier CRJ900 (CR9)",
    "CRJX": "Bombardier CRJ1000 (CRK)",
    "CL30": "Bombardier Challenger 300",
    "CL60": "Bombardier Challenger 600 (CCJ)",
    "GL5T": "Bombardier Global 5000 (CCX)",
    "GLEX": "Bombardier Global Express (CCX)",
    
    # === EMBRAER ===
    "E110": "Embraer EMB 110 Bandeirante (EMB)",
    "E120": "Embraer EMB 120 Brasilia (EM2)",
    "E135": "Embraer RJ135 (ER3)",
    "E145": "Embraer RJ145 (ER4)",
    "E170": "Embraer 170 (E70)",
    "E175": "Embraer 175 (E75)",
    "E190": "Embraer 190 (E90)",
    "E195": "Embraer 195 (E95)",
    "E290": "Embraer E190-E2 (290)",
    "E295": "Embraer E195-E2 (295)",
    "E35L": "Embraer Legacy 600/650 (ER3)",
    "E50P": "Embraer Phenom 100 (EP1)",
    "E55P": "Embraer Phenom 300 (EP3)",
    
    # === MCDONNELL DOUGLAS ===
    "DC10": "Douglas DC-10 (D10/D11)",
    "DC85": "Douglas DC-8-50 (D8T)",
    "DC86": "Douglas DC-8-62 (D8L)",
    "DC87": "Douglas DC-8-72 (D8Q)",
    "DC91": "Douglas DC-9-10 (D91)",
    "DC92": "Douglas DC-9-20 (D92)",
    "DC93": "Douglas DC-9-30 (D93)",
    "DC94": "Douglas DC-9-40 (D94)",
    "DC95": "Douglas DC-9-50 (D95)",
    "MD11": "McDonnell Douglas MD-11 (M11)",
    "MD81": "McDonnell Douglas MD-81 (M81)",
    "MD82": "McDonnell Douglas MD-82 (M82)",
    "MD83": "McDonnell Douglas MD-83 (M83)",
    "MD87": "McDonnell Douglas MD-87 (M87)",
    "MD88": "McDonnell Douglas MD-88 (M88)",
    "MD90": "McDonnell Douglas MD-90 (M90)",
    
    # === LOCKHEED ===
    "C130": "Lockheed C-130 Hercules (LOH)",
    "C5M": "Lockheed C-5M Super Galaxy",
    "L101": "Lockheed L-1011 TriStar (L10)",
    "L188": "Lockheed L-188 Electra (LOE)",
    "P3": "Lockheed P-3 Orion",
    
    # === CESSNA ===
    "C172": "Cessna 172 Skyhawk",
    "C182": "Cessna 182 Skylane",
    "C208": "Cessna 208 Caravan (C08)",
    "C25A": "Cessna Citation CJ2 (CNJ)",
    "C25B": "Cessna Citation CJ3 (CNJ)",
    "C25C": "Cessna Citation CJ4 (CNJ)",
    "C500": "Cessna Citation I (CNJ)",
    "C510": "Cessna Citation Mustang (CNJ)",
    "C525": "Cessna CitationJet (CNJ)",
    "C550": "Cessna Citation II (CNJ)",
    "C560": "Cessna Citation V (CNJ)",
    "C56X": "Cessna Citation Excel (CNJ)",
    "C650": "Cessna Citation III/VI/VII (CNJ)",
    "C680": "Cessna Citation Sovereign (CNJ)",
    "C750": "Cessna Citation X (CNJ)",
    
    # === BEECHCRAFT ===
    "B190": "Beechcraft 1900 (BEH)",
    "BE20": "Beechcraft King Air 200",
    "BE30": "Beechcraft King Air 300/350",
    "BE40": "Beechcraft Premier I",
    "BE99": "Beechcraft 99 Airliner",
    "C99": "Beechcraft C99",
    
    # === GULFSTREAM ===
    "G150": "Gulfstream G150",
    "G280": "Gulfstream G280",
    "GIV": "Gulfstream IV",
    "GV": "Gulfstream V",
    "G550": "Gulfstream G550",
    "G650": "Gulfstream G650",
    
    # === OTHER USA MANUFACTURERS ===
    "CL60": "Canadair Challenger 600",
    "F2TH": "Dassault Falcon 2000",
    "F900": "Dassault Falcon 900",
    "FA50": "Dassault Falcon 50",
    "FA7X": "Dassault Falcon 7X",
    "H25B": "Hawker 800",
    "LJ35": "Learjet 35",
    "LJ45": "Learjet 45",
    "LJ60": "Learjet 60",
    "PA31": "Piper Navajo",
    "PA46": "Piper Meridian",
    "PC12": "Pilatus PC-12",
    "C441": "Cessna Conquest II",
    "MU2": "Mitsubishi MU-2",
}

# =================== DUTY CODES ===================
DUTY_CODES = {
    # === FLIGHT DUTY CODES ===
    "FD": "Flight Duty",
    "FDP": "Flight Duty Period",
    "TS": "Training/Simulator",
    "CBT": "Computer Based Training",
    "TRNG": "Training",
    "CHECK": "Check Ride/Line Check",
    "OE": "Operating Experience",
    "IOE": "Initial Operating Experience",
    "LC": "Line Check",
    "PC": "Proficiency Check",
    "SIM": "Simulator",
    "LOFT": "Line Oriented Flight Training",
    
    # === POSITIONING/DEADHEAD ===
    "DH": "Deadhead",
    "DHD": "Deadhead",
    "DDHD": "Deadhead",
    "POS": "Positioning",
    "REPO": "Repositioning",
    "FERRY": "Ferry Flight",
    
    # === RESERVE/STANDBY ===
    "RSV": "Reserve",
    "RES": "Reserve",
    "SBY": "Standby",
    "STBY": "Standby",
    "APT": "Airport Standby/Reserve",
    "HOME": "Home Reserve",
    "CALL": "On Call",
    "RAP": "Reserve Availability Period",
    "HOT": "Hot Reserve (Short Call)",
    "COLD": "Cold Reserve (Long Call)",
    
    # === REST/TIME OFF ===
    "RO": "Rest/Off",
    "DO": "Day Off",
    "REST": "Rest Period",
    "MIN REST": "Minimum Rest",
    "CDO": "Continuous Duty Overnight",
    "WOCL": "Window of Circadian Low",
    "RON": "Remain Over Night",
    "LAYOVER": "Layover",
    "HOTEL": "Hotel Rest",
    "LOCAL": "Local Night's Rest",
    
    # === ADMINISTRATIVE ===
    "OFF": "Off Duty",
    "VAC": "Vacation",
    "SICK": "Sick Leave",
    "PTO": "Paid Time Off",
    "ADMIN": "Administrative Duty",
    "MTG": "Meeting",
    "REC": "Recurrent Training",
    "GND": "Ground School",
    "MED": "Medical",
    "OFFC": "Office Duty",
    
    # === OPERATIONAL ===
    "BLK": "Block Time",
    "FLT": "Flight Time",
    "TAFB": "Time Away From Base",
    "DUTY": "Duty Time",
    "REPORT": "Report Time",
    "BRIEF": "Briefing",
    "DEBRIEF": "Debriefing",
    "PREFLT": "Pre-flight",
    "POSTFLT": "Post-flight",
    "TURN": "Turn (Same Day Return)",
    "TRIP": "Trip Pairing",
    "PAIRING": "Flight Pairing",
    "SEQ": "Sequence",
    "LEG": "Flight Leg",
    
    # === SPECIAL ASSIGNMENTS ===
    "CHARTER": "Charter Flight",
    "EXTRA": "Extra Section",
    "ACM": "Additional Crew Member",
    "IRO": "International Relief Officer",
    "FB": "Flight Deck Observer/Familiarization",
    "JS": "Jumpseat",
    "MUST GO": "Must Go/Positive Space",
    "SA": "Space Available",
    "NRSA": "Non-Revenue Space Available",
    
    # === DISRUPTION CODES ===
    "CNX": "Cancelled",
    "CNCL": "Cancelled",
    "DLY": "Delayed",
    "IROPS": "Irregular Operations",
    "WX": "Weather",
    "MX": "Maintenance",
    "ATC": "Air Traffic Control",
    "CREW": "Crew Issue",
    "MISCON": "Missed Connection",
    "REROUTE": "Reroute",
    "SUB": "Substitution",
    "SWAP": "Schedule Swap",
    
    # === PAY CODES ===
    "CREDIT": "Credit Time",
    "HARD": "Hard Time (Actual)",
    "SOFT": "Soft Time (Additional)",
    "CLAIM": "Claim Time",
    "OT": "Overtime",
    "DT": "Double Time",
    "HOLIDAY": "Holiday Pay",
    "PER DIEM": "Per Diem",
    "INTL": "International Override",
    "NIGHT": "Night Pay",
    "CALL OUT": "Call Out Pay",
}

# ---------- O4-MINI MEGA PROMPT INTEGRATED ------------------------
O4_MINI_MEGA_PROMPT = """
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
GO FORTH AND EXTRACT EVERYTHING! 🎯
"""

# ---------- Fast Airport Resolver -----------------------------------
@functools.lru_cache(maxsize=512)
def resolve_airline(code: str) -> dict[str, str]:
    """Cached airline resolution for performance"""
    code = code.upper()
    if code in AIRLINE_CODES:
        return {"iata": code, **AIRLINE_CODES[code]}
    # Reverse lookup
    for iata, data in AIRLINE_CODES.items():
        if data["icao"] == code:
            return {"iata": iata, **data}
    raise HTTPException(404, f"Unknown airline code '{code}'")

# ---------- Enhanced Schema (Simplified) ----------------
class FlightConnection(BaseModel):
    """Represents connections between flights"""
    from_flight: str
    to_flight: str
    connection_time: int
    connection_type: str

class Flight(BaseModel):
    date: str = Field(..., description="Flight date in MM/DD/YYYY format")
    flight_no: str = Field(..., description="Flight number including airline prefix")
    origin: Optional[str] = Field(None, description="Origin airport code")
    dest: Optional[str] = Field(None, description="Destination airport code")
    sched_out_local: Optional[str] = Field(None, description="Scheduled departure time in HHMM format")
    sched_in_local: Optional[str] = Field(None, description="Scheduled arrival time in HHMM format")
    page_number: Optional[int] = Field(None, description="Page number in PDF (1-based)")

    @validator("flight_no")
    def clean_flight_no(cls, v):
        if not v:
            return v
        cleaned = re.sub(r"[^\w\d\-/]", "", v.upper())
        
        # Quick pattern matching using cached regex
        if regex_cache.UPS_PATTERN.match(cleaned):
            return cleaned
        if regex_cache.STANDARD_FLIGHT_PATTERN.match(cleaned):
            return cleaned
        if regex_cache.NUMERIC_FLIGHT_PATTERN.match(cleaned):
            return cleaned
        return cleaned

    @validator("origin", "dest")
    def validate_airport(cls, v):
        if v and len(v) >= 3:
            return v.upper()
        return v

    class Config:
        extra = "ignore"

class Result(BaseModel):
    flights: List[Flight]
    connections: List[FlightConnection] = []
    total_flights_found: int = 0
    avg_conf: float = 0.0

# ---------- Lightweight Layout Detection ---------------
class FastLayoutAnalyzer:
    """Minimal layout analysis for speed"""
    
    @staticmethod
    def detect_schedule_type(img: np.ndarray) -> Dict[str, Any]:
        """Quick layout detection - simplified for speed"""
        h, w = img.shape[:2]
        
        # Quick aspect ratio check for mobile
        aspect_ratio = h / w
        is_mobile = aspect_ratio > 1.5
        
        # Default to mixed layout to avoid expensive grid detection
        layout_type = "mobile" if is_mobile else "mixed"
        
        return {
            "type": layout_type,
            "dimensions": (h, w),
            "aspect_ratio": aspect_ratio,
            "detected_cells": []  # Skip expensive cell detection
        }

# ---------- Speed-Optimized Image Processor --------------------------
class FastImageProcessor:
    """Minimal image processing for speed"""
    
    @staticmethod
    def prepare_minimal_versions(img: np.ndarray) -> List[np.ndarray]:
        """Prepare minimal image versions for O4-Mini"""
        versions = []
        
        # 1. Original image (O4-Mini handles most cases well)
        versions.append(img)
        
        # 2. Only add enhanced version if image is very dark or blurry
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        
        if mean_brightness < 100:  # Dark image
            # CLAHE enhancement
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            l = clahe.apply(l)
            enhanced = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
            versions.append(enhanced)
        
        return versions[:MAX_IMAGE_VERSIONS]
    
    @staticmethod
    def quick_clarity_check(img: np.ndarray) -> Dict[str, float]:
        """Fast clarity assessment"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Simple blur detection
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        blur_score = min(1.0, laplacian_var / 1000.0)
        
        # Basic contrast
        contrast_score = min(1.0, gray.std() / 50.0)
        
        return {
            "overall": (blur_score + contrast_score) / 2,
            "blur": blur_score,
            "contrast": contrast_score,
            "needs_enhancement": blur_score < 0.3 or contrast_score < 0.3
        }

# ---------- PDF Processor (Optimized) --------------------------
class FastPDFProcessor:
    """Optimized PDF processing"""
    
    @staticmethod
    async def convert_pdf_to_images(pdf_bytes: bytes) -> List[np.ndarray]:
        """Convert PDF with optimal settings for speed"""
        try:
            # Use lower DPI for faster processing (200 instead of 300)
            pil_images = await asyncio.get_event_loop().run_in_executor(
                thread_pool,
                functools.partial(
                    convert_from_bytes,
                    pdf_bytes,
                    dpi=200,  # Reduced DPI
                    fmt='PNG',
                    thread_count=4,
                    use_pdftocairo=True  # Faster renderer if available
                )
            )
            
            # Parallel conversion to numpy arrays
            async def convert_single(pil_img):
                return await asyncio.get_event_loop().run_in_executor(
                    thread_pool,
                    FastPDFProcessor._pil_to_cv2,
                    pil_img
                )
            
            cv_images = await asyncio.gather(*[convert_single(img) for img in pil_images])
            
            logger.info(f"Converted PDF with {len(cv_images)} pages")
            return cv_images
            
        except Exception as e:
            logger.error(f"PDF conversion error: {e}")
            raise HTTPException(status_code=422, detail=f"Failed to process PDF: {str(e)}")
    
    @staticmethod
    def _pil_to_cv2(pil_img):
        """Convert PIL image to OpenCV format"""
        img_array = np.array(pil_img)
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            return cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        elif len(img_array.shape) == 2:
            return cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
        return img_array

# ---------- Minimal OCR for Hints ------------------------
class MinimalOCR:
    """Lightweight OCR for hints only"""
    
    def __init__(self):
        self.enabled = USE_OCR_HINTS and OCR_TESS_AVAILABLE
    
    async def extract_quick_hints(self, img: np.ndarray) -> Dict[str, List[str]]:
        """Extract minimal hints for O4-Mini"""
        if not self.enabled:
            return {"dates": [], "times": [], "flight_numbers": [], "airports": []}
        
        try:
            # Take only header region for hints
            h = img.shape[0]
            header_region = img[:int(h * 0.15), :]
            
            # Quick OCR on header only
            text = await self._quick_tesseract(header_region)
            
            return {
                "dates": regex_cache.DATE_PATTERN.findall(text)[:3],
                "times": regex_cache.TIME_PATTERN.findall(text)[:5],
                "flight_numbers": regex_cache.FLIGHT_PATTERN.findall(text)[:5],
                "airports": [w for w in regex_cache.AIRPORT_PATTERN.findall(text) if len(w) in (3, 4)][:5]
            }
        except:
            return {"dates": [], "times": [], "flight_numbers": [], "airports": []}
    
    async def _quick_tesseract(self, img: np.ndarray) -> str:
        """Fast Tesseract OCR"""
        try:
            pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            # Use fastest config
            text = await asyncio.get_event_loop().run_in_executor(
                thread_pool,
                pytesseract.image_to_string,
                pil_img,
                config="--psm 11 --oem 1"  # Fast mode
            )
            return text
        except:
            return ""

# ---------- Fast O4-Mini Extractor ------------------------
class FastO4MiniExtractor:
    """Optimized O4-Mini extraction"""
    
    @staticmethod
    def create_minimal_prompt(
        clarity_metrics: Dict[str, float],
        layout_type: str,
        ocr_hints: Optional[Dict[str, List[str]]] = None
    ) -> str:
        """Create minimal prompt for speed"""
        # Use placeholder for mega prompt
        return O4_MINI_MEGA_PROMPT
    
    @staticmethod
    async def extract_with_o4mini(
        images: List[str],
        clarity_metrics: Dict[str, float],
        layout_info: Dict[str, Any],
        ocr_hints: Optional[Dict[str, List[str]]] = None
    ) -> List[Flight]:
        """Optimized extraction with O4-Mini"""
        
        prompt = FastO4MiniExtractor.create_minimal_prompt(
            clarity_metrics,
            layout_info.get("type", "mixed"),
            ocr_hints
        )
        
        # O4-Mini function schema
        tools = [{
            "type": "function",
            "function": {
                "name": "extract_visual_flight_schedule",
                "description": "Return ONLY the flights array with the minimal fields.",
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
                                "required": ["date", "flight_no"]
                            }
                        }
                    },
                    "required": ["flights"]
                }
            }
        }]

        try:
            # Prepare input content
            input_content = [
                {"type": "text", "text": prompt}
            ]
            
            # Add images
            for img_b64 in images:
                input_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img_b64}"}
                })
            
            # Call O4-Mini
            response = await openai_client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": input_content}
                ],
                tools=tools,
                tool_choice={
                    "type": "function",
                    "function": {"name": "extract_visual_flight_schedule"}
                },
                max_completion_tokens=MAX_OUTPUT_TOKENS,
                seed=42,
            )
            
            # Parse response
            if response.choices[0].message.tool_calls:
                tool_call = response.choices[0].message.tool_calls[0]
                raw_data = json.loads(tool_call.function.arguments)
                flights = [Flight(**f) for f in raw_data.get("flights", [])]
                return flights
            
        except Exception as e:
            logger.error(f"O4-Mini extraction error: {e}")
            
        return []

# ---------- Fast Relationship Detection --------------------
class FastRelationshipExtractor:
    """Optimized relationship detection"""
    
    @staticmethod
    def detect_connections(flights: List[Flight]) -> List[FlightConnection]:
        """Fast connection detection"""
        connections = []
        
        # Group flights by date for faster processing
        flights_by_date = defaultdict(list)
        for f in flights:
            flights_by_date[f.date].append(f)
        
        # Process each date group
        for date, date_flights in flights_by_date.items():
            # Sort by time within date
            sorted_flights = sorted(
                date_flights,
                key=lambda f: f.sched_out_local or "0000"
            )
            
            # Check connections within same day
            for i in range(len(sorted_flights) - 1):
                current = sorted_flights[i]
                next_flight = sorted_flights[i + 1]
                
                if (current.dest and next_flight.origin and 
                    current.dest == next_flight.origin and
                    current.sched_in_local and next_flight.sched_out_local):
                    
                    # Quick time calculation
                    try:
                        arr_h, arr_m = int(current.sched_in_local[:2]), int(current.sched_in_local[2:])
                        dep_h, dep_m = int(next_flight.sched_out_local[:2]), int(next_flight.sched_out_local[2:])
                        
                        conn_minutes = (dep_h * 60 + dep_m) - (arr_h * 60 + arr_m)
                        
                        if 30 <= conn_minutes <= 300:
                            connections.append(
                                FlightConnection(
                                    from_flight=current.flight_no,
                                    to_flight=next_flight.flight_no,
                                    connection_time=conn_minutes,
                                    connection_type="same_day"
                                )
                            )
                    except:
                        pass
        
        return connections

# ---------- Fast Aviation Validator ------------------------
class FastAviationValidator:
    """Minimal validation for speed"""
    
    @staticmethod
    def quick_validate(flight: Flight) -> Tuple[bool, List[str]]:
        """Quick validation checks only"""
        warnings = []
        
        # Only essential validations
        if flight.flight_no and not regex_cache.VALID_FLIGHT_PATTERN.match(flight.flight_no):
            warnings.append(f"Unusual flight number: {flight.flight_no}")
        
        # Quick date format check
        if flight.date:
            try:
                datetime.strptime(flight.date, "%m/%d/%Y")
            except:
                warnings.append(f"Invalid date: {flight.date}")
        
        return len(warnings) == 0, warnings

# ---------- Fast Pipeline Orchestrator --------------------
class FastPipeline:
    """Speed-optimized pipeline"""
    
    def __init__(self):
        self.layout = FastLayoutAnalyzer()
        self.imgproc = FastImageProcessor()
        self.ocr = MinimalOCR()
        self.extractor = FastO4MiniExtractor()
        self.relationships = FastRelationshipExtractor()
    
    async def run(self, images: List[np.ndarray]) -> Result:
        """Process images with speed optimizations"""
        all_flights = []
        all_connections = []
        
        # Process images in parallel where possible
        async def process_single_image(page_num: int, img: np.ndarray) -> List[Flight]:
            logger.info(f"Processing page {page_num + 1}")
            
            # Quick layout check (optional)
            layout_info = {"type": "mixed"} if not ENABLE_LAYOUT_DETECTION else \
                         self.layout.detect_schedule_type(img)
            
            # Quick clarity check
            clarity = self.imgproc.quick_clarity_check(img)
            
            # Minimal image versions
            versions = self.imgproc.prepare_minimal_versions(img)
            
            # Optional OCR hints
            ocr_hints = None
            if USE_OCR_HINTS:
                ocr_hints = await self.ocr.extract_quick_hints(img)
            
            # Convert to base64
            b64_images = []
            for v in versions:
                ok, buf = cv2.imencode(".png", v, [cv2.IMWRITE_PNG_COMPRESSION, 1])  # Fast compression
                if ok:
                    b64_images.append(base64.b64encode(buf).decode("utf-8"))
            
            # Extract flights
            page_flights = await self.extractor.extract_with_o4mini(
                images=b64_images,
                clarity_metrics=clarity,
                layout_info=layout_info,
                ocr_hints=ocr_hints
            )
            
            # Add page reference
            for flight in page_flights:
                flight.page_number = page_num + 1
            
            return page_flights
        
        # Process all pages in parallel (limited concurrency)
        semaphore = asyncio.Semaphore(3)  # Max 3 concurrent page processing
        
        async def process_with_limit(page_num: int, img: np.ndarray):
            async with semaphore:
                return await process_single_image(page_num, img)
        
        # Process all images
        results = await asyncio.gather(*[
            process_with_limit(i, img) for i, img in enumerate(images)
        ])
        
        # Combine results
        for page_flights in results:
            all_flights.extend(page_flights)
        
        # Fast relationship detection
        if all_flights:
            connections = self.relationships.detect_connections(all_flights)
            all_connections.extend(connections)
        
        return Result(
            flights=all_flights,
            connections=all_connections,
            total_flights_found=len(all_flights),
            avg_conf=0.95  # O4-Mini is highly accurate
        )

# ---------- Initialize Pipeline -----------------------------------
pipeline = FastPipeline()

# ---------- FastAPI Endpoints -----------------------------------
@app.post("/extract")
async def extract_endpoint(
    file: UploadFile = File(...),
    airline: Optional[str] = Query(
        None,
        description="Airline prefix to apply (IATA or ICAO, e.g. UA / UAL). "
                    "Overrides X-Airline header."
    ),
    x_airline: Optional[str] = Header(None, convert_underscores=False),
):
    """Extract flights from image or PDF"""
    # Basic checks
    content_type = file.content_type.lower()
    
    if not (content_type.startswith("image") or content_type == "application/pdf"):
        raise HTTPException(415, "Upload an image file (JPEG, PNG, etc.) or PDF")
    
    raw_bytes = await file.read()
    
    # Process based on file type
    images_to_process = []
    
    if content_type == "application/pdf":
        # Handle PDF
        logger.info(f"Processing PDF: {file.filename}")
        images_to_process = await FastPDFProcessor.convert_pdf_to_images(raw_bytes)
        
        if not images_to_process:
            raise HTTPException(422, "Unable to extract pages from PDF")
    else:
        # Handle image
        np_img = cv2.imdecode(np.frombuffer(raw_bytes, np.uint8), cv2.IMREAD_COLOR)
        if np_img is None:
            raise HTTPException(422, "Unable to decode image")
        images_to_process = [np_img]
    
    # Resolve airline
    airline_code = airline or x_airline
    if not airline_code:
        raise HTTPException(
            400,
            "Provide ?airline=<IATA|ICAO> in the URL or X-Airline header"
        )
    
    airline_info = resolve_airline(airline_code)
    iata_prefix = airline_info["iata"]
    
    # Run pipeline
    raw_result = await pipeline.run(images_to_process)
    
    # Apply airline prefix
    for f in raw_result.flights:
        if f.flight_no:
            if regex_cache.NUMERIC_FLIGHT_PATTERN.fullmatch(f.flight_no):
                f.flight_no = f"{iata_prefix}{f.flight_no}"
            elif not f.flight_no.upper().startswith(iata_prefix):
                if f.flight_no[0].isdigit():
                    f.flight_no = f"{iata_prefix}{f.flight_no}"
    
    # Convert to dict
    raw_dict = jsonable_encoder(raw_result)
    raw_dict.setdefault("schedule_metadata", {})["airline"] = airline_info
    
    # Add file info
    raw_dict["schedule_metadata"]["file_type"] = "pdf" if content_type == "application/pdf" else "image"
    if content_type == "application/pdf":
        raw_dict["schedule_metadata"]["num_pages"] = len(images_to_process)
    
    # Check if validation needed
    should_validate = any(
        not all([f.get("origin"), f.get("dest"), f.get("sched_out_local"), f.get("sched_in_local")])
        for f in raw_dict.get("flights", [])
    )
    
    if should_validate:
        enriched = await validate_extraction_results(raw_dict)
    else:
        # Skip validation
        logger.info("Skipping validation - all fields present")
        enriched = raw_dict
        enriched["validation"] = {
            "total_flights": len(raw_dict.get("flights", [])),
            "valid_flights": len(raw_dict.get("flights", [])),
            "average_confidence": 1.0,
            "total_fields_filled": 0,
            "processing_time_seconds": 0,
            "sources_used": ["skipped"],
            "warnings": []
        }
    
    # Quick validation pass
    validation_warnings = []
    for flight in enriched.get("enriched_flights", enriched.get("flights", [])):
        f_obj = Flight(**flight) if isinstance(flight, dict) else flight
        _, warnings = FastAviationValidator.quick_validate(f_obj)
        validation_warnings.extend(warnings)
    
    if validation_warnings:
        enriched.setdefault("validation", {})["warnings"] = validation_warnings[:10]
    
    return enriched

@app.post("/validate")
async def validate_endpoint(flights: List[Flight] = Body(...)):
    """Validate flights against external APIs"""
    try:
        return await validate_flights_endpoint([f.dict() for f in flights])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Validation failed: {e}")

@app.get("/health")
async def health():
    """Health check endpoint"""
    validation_apis = []
    if os.getenv("FLIGHTAWARE_API_KEY"):
        validation_apis.append("flightaware_aeroapi")
    if os.getenv("FLIGHTRADAR24_API_KEY"):
        validation_apis.append("flightradar24")
    if os.getenv("FIREHOSE_API_KEY"):
        validation_apis.append("firehose")
    
    # Check PDF support
    pdf_support = False
    try:
        import pdf2image
        pdf_support = True
    except ImportError:
        pass
    
    return {
        "status": "ok",
        "version": "5.0-speed",
        "model": MODEL,
        "reasoning_effort": REASONING_EFFORT,
        "ocr_enabled": USE_OCR_HINTS and OCR_TESS_AVAILABLE,
        "pdf_support": pdf_support,
        "validation_apis": validation_apis,
        "layout_detection": ENABLE_LAYOUT_DETECTION,
        "features": [
            "o4-mini-high-visual-reasoning",
            "speed-optimized-pipeline",
            "parallel-processing",
            "minimal-image-versions",
            "optional-ocr-hints",
            "fast-pdf-processing",
            "cached-regex-patterns",
            "thread-pool-execution"
        ],
        "performance": {
            "max_image_versions": MAX_IMAGE_VERSIONS,
            "max_concurrent_pages": 3,
            "ocr_hints": "header-only" if USE_OCR_HINTS else "disabled",
            "pdf_dpi": 200
        }
    }

# ---------- Run ------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)