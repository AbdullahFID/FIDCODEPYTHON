"""
Enhanced Flight-Intel UPS Roster Extractor v4.0
- Multi-model ensemble architecture
- Advanced layout detection and classification
- Enhanced OCR with multiple engines
- Contextual extraction with relationship detection
- Domain-specific validation
- Target: 75%+ accuracy across all formats
"""
# uvicorn main:app --reload
from dotenv import load_dotenv
load_dotenv()
import os, base64, cv2, numpy as np, re, json, asyncio, hashlib, pickle
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Any
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from openai import AsyncOpenAI
from dateutil import parser
import logging
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict, Counter
import Levenshtein
from flight_intel_patch import (
    validate_extraction_results,
    validate_flights_endpoint,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------- Enhanced OCR imports ------------------------------
# ---------- Enhanced OCR imports ------------------------------
OCR_AVAILABLE = False
try:
    import pytesseract, easyocr
    from PIL import Image, ImageEnhance, ImageFilter, ImageOps
    if not hasattr(Image, "ANTIALIAS"):
        Image.ANTIALIAS = Image.Resampling.LANCZOS


    # 1️⃣  EasyOCR – force CPU on Apple Silicon
    try:
        easy_reader = easyocr.Reader(["en"], gpu=False)
        OCR_AVAILABLE = True
    except Exception:
        easy_reader = None


except ImportError:
    OCR_AVAILABLE = False
    logger.warning(
        "OCR libraries not available. Install pytesseract, easyocr, and paddleocr for better results."
    )

# ---------- Configuration ------------------------------------
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("MODEL_NAME", "gpt-4o")
TEMP = float(os.getenv("OPENAI_TEMPERATURE", 0.1))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", 8192))

if not OPENAI_KEY:
    raise RuntimeError("OPENAI_API_KEY missing")

openai_client = AsyncOpenAI(api_key=OPENAI_KEY)

app = FastAPI(title="Flight-Intel Vision API", version="4.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# ---------- Aviation Domain Knowledge -------------------------

# =================== AIRLINE CODES ===================
# Format: IATA Code: Full Name (ICAO Code if different)

AIRLINE_CODES = {
    # === US CARRIERS ===
    "AA": "American Airlines",
    "UA": "United Airlines", 
    "DL": "Delta Air Lines",
    "WN": "Southwest Airlines",
    "B6": "JetBlue Airways",
    "AS": "Alaska Airlines",
    "NK": "Spirit Airlines",
    "F9": "Frontier Airlines",
    "G4": "Allegiant Air",
    "SY": "Sun Country Airlines",
    "HA": "Hawaiian Airlines",
    "MX": "Breeze Airways",
    
    # === CARGO CARRIERS ===
    "5X": "UPS Airlines (UPS)",
    "FX": "FedEx Express (FDX)",
    "5Y": "Atlas Air",
    "K4": "Kalitta Air",
    "NC": "Northern Air Cargo",
    "GB": "ABX Air",
    "3S": "Polar Air Cargo",
    "M6": "Amerijet International",
    "6R": "AeroUnion",
    "CV": "Cargolux",
    "CK": "China Cargo Airlines",
    "8Y": "Silk Way West Airlines",
    "PO": "Polar Air Cargo",
    "KZ": "Nippon Cargo Airlines",
    "OO": "SkyWest Airlines",
    "9E": "Endeavor Air",
    "ZW": "Air Wisconsin",
    "YV": "Mesa Airlines",
    "YX": "Republic Airways",
    "OH": "PSA Airlines",
    "MQ": "Envoy Air",
    "G7": "GoJet Airlines",
    "CP": "Compass Airlines",
    "PT": "Piedmont Airlines",
    
    # === CANADIAN CARRIERS ===
    "AC": "Air Canada",
    "WS": "WestJet",
    "TS": "Air Transat",
    "PD": "Porter Airlines",
    "F8": "Flair Airlines",
    "Y9": "Kenn Borek Air",
    "ZX": "Air Georgian",
    "QK": "Jazz Aviation",
    
    # === EUROPEAN CARRIERS ===
    "BA": "British Airways",
    "LH": "Lufthansa",
    "AF": "Air France",
    "KL": "KLM Royal Dutch Airlines",
    "IB": "Iberia",
    "AZ": "ITA Airways (formerly Alitalia)",
    "LX": "Swiss International Air Lines",
    "OS": "Austrian Airlines",
    "SN": "Brussels Airlines",
    "SK": "Scandinavian Airlines (SAS)",
    "AY": "Finnair",
    "TP": "TAP Air Portugal",
    "LO": "LOT Polish Airlines",
    "OK": "Czech Airlines",
    "RO": "TAROM",
    "A3": "Aegean Airlines",
    "BT": "airBaltic",
    "OU": "Croatia Airlines",
    "JU": "Air Serbia",
    "FB": "Bulgaria Air",
    "PS": "Ukraine International Airlines",
    
    # === EUROPEAN LOW-COST ===
    "FR": "Ryanair",
    "U2": "easyJet",
    "W6": "Wizz Air",
    "VY": "Vueling",
    "DY": "Norwegian Air Shuttle",
    "HV": "Transavia",
    "PC": "Pegasus Airlines",
    "0B": "Blue Air",
    "W9": "Wizz Air UK",
    "EW": "Eurowings",
    "LS": "Jet2",
    "BE": "Flybe",
    
    # === MIDDLE EASTERN CARRIERS ===
    "EK": "Emirates",
    "QR": "Qatar Airways",
    "EY": "Etihad Airways",
    "SV": "Saudia",
    "GF": "Gulf Air",
    "WY": "Oman Air",
    "KU": "Kuwait Airways",
    "MS": "EgyptAir",
    "ME": "Middle East Airlines",
    "RJ": "Royal Jordanian",
    "FZ": "flydubai",
    "G9": "Air Arabia",
    "TK": "Turkish Airlines",
    "XY": "flynas",
    "J9": "Jazeera Airways",
    "6E": "IndiGo",
    
    # === ASIAN CARRIERS ===
    "CX": "Cathay Pacific",
    "SQ": "Singapore Airlines",
    "TG": "Thai Airways",
    "MH": "Malaysia Airlines",
    "GA": "Garuda Indonesia",
    "PR": "Philippine Airlines",
    "VN": "Vietnam Airlines",
    "JL": "Japan Airlines",
    "NH": "All Nippon Airways",
    "OZ": "Asiana Airlines",
    "KE": "Korean Air",
    "BR": "EVA Air",
    "CI": "China Airlines",
    "CA": "Air China",
    "CZ": "China Southern Airlines",
    "MU": "China Eastern Airlines",
    "HU": "Hainan Airlines",
    "FM": "Shanghai Airlines",
    "3U": "Sichuan Airlines",
    "ZH": "Shenzhen Airlines",
    "SC": "Shandong Airlines",
    "BX": "Air Busan",
    "7C": "Jeju Air",
    "TW": "T'way Air",
    "LJ": "Jin Air",
    "AI": "Air India",
    "SG": "SpiceJet",
    "UK": "Vistara",
    "IX": "Air India Express",
    "QZ": "AirAsia Indonesia",
    "AK": "AirAsia",
    "D7": "AirAsia X",
    "FD": "Thai AirAsia",
    "VJ": "VietJet Air",
    "PG": "Bangkok Airways",
    "DD": "Nok Air",
    "SL": "Thai Lion Air",
    "JQ": "Jetstar Airways",
    "TR": "Scoot",
    "3K": "Jetstar Asia",
    "GK": "Jetstar Japan",
    "MM": "Peach Aviation",
    "BC": "Skymark Airlines",
    "NU": "Japan Transocean Air",
    "HD": "Air Do",
    "UO": "Hong Kong Express",
    "BL": "Jetstar Pacific",
    "VZ": "Thai Vietjet Air",
    
    # === AFRICAN CARRIERS ===
    "ET": "Ethiopian Airlines",
    "SA": "South African Airways",
    "KQ": "Kenya Airways",
    "MS": "EgyptAir",
    "AT": "Royal Air Maroc",
    "TU": "Tunisair",
    "AH": "Air Algérie",
    "SW": "Air Namibia",
    "MK": "Air Mauritius",
    "MD": "Air Madagascar",
    "TC": "Air Tanzania",
    "KP": "ASKY Airlines",
    "SS": "Corsair",
    
    # === LATIN AMERICAN CARRIERS ===
    "LA": "LATAM Airlines",
    "CM": "Copa Airlines",
    "AV": "Avianca",
    "AM": "Aeroméxico",
    "AR": "Aerolíneas Argentinas",
    "G3": "Gol Linhas Aéreas",
    "AD": "Azul Brazilian Airlines",
    "4M": "LATAM Argentina",
    "JJ": "LATAM Brasil",
    "4C": "LATAM Colombia",
    "XL": "LATAM Ecuador",
    "PZ": "LATAM Paraguay",
    "LP": "LATAM Peru",
    "2Z": "Viva Aerobus",
    "Y4": "Volaris",
    "VB": "VivaAerobus",
    "5U": "TAG Airlines",
    "TA": "TACA",
    "P9": "Peruvian Airlines",
    "H2": "Sky Airline",
    "JA": "JetSMART",
    "5J": "Cebu Pacific",
    
    # === OCEANIA CARRIERS ===
    "QF": "Qantas",
    "VA": "Virgin Australia",
    "NZ": "Air New Zealand",
    "FJ": "Fiji Airways",
    "QN": "Air Armenia",
    
    # === CHARTER/LEISURE CARRIERS ===
    "X3": "TUI fly",
    "BY": "TUI Airways",
    "OR": "TUI fly Netherlands",
    "TB": "TUI fly Belgium",
    "DE": "Condor",
    "LS": "Jet2",
    "MT": "Thomas Cook Airlines",
    
    # === REGIONAL CARRIERS ===
    "ZL": "Regional Express",
    "PB": "Provincial Airlines",
    "8J": "Kargo Xpress",
    "XT": "Xtra Airways",
    
    # === OTHER NOTABLE CARRIERS ===
    "SU": "Aeroflot",
    "S7": "S7 Airlines",
    "UN": "Transaero Airlines",
    "FV": "Rossiya Airlines",
    "UT": "UTair",
    "R3": "Yakutia Airlines",
    "KC": "Air Astana",
    "HY": "Uzbekistan Airways",
    "DV": "SCAT Airlines",
    "7R": "RusLine",
    "LY": "El Al Israel Airlines",
    "4X": "Mercury Air Cargo",
    "6A": "Armenia Airways",
}

# =================== AIRPORT CODES ===================
# Major airports worldwide (IATA codes)

MAJOR_AIRPORTS = {
    # === NORTH AMERICA - USA ===
    # Major Hubs
    "ATL", "ORD", "LAX", "DFW", "DEN", "JFK", "SFO", "SEA", "LAS", "MCO",
    "PHX", "EWR", "IAH", "MIA", "BOS", "MSP", "DTW", "FLL", "PHL", "LGA",
    "CLT", "BWI", "DCA", "SLC", "SAN", "TPA", "MDW", "BNA", "AUS", "PDX",
    "STL", "MCI", "OAK", "SMF", "SJC", "RDU", "RSW", "CLE", "CVG", "IND",
    "MKE", "PIT", "CMH", "SAT", "MSY", "JAX", "RIC", "BDL", "ALB", "BUF",
    "SYR", "ROC", "PVD", "MHT", "PWM", "BTV", "BGR", "ORF", "RNO", "ABQ",
    "TUS", "ELP", "OKC", "TUL", "XNA", "LIT", "DSM", "OMA", "ICT", "CID",
    "FAR", "GRR", "FNT", "SBN", "EVV", "LEX", "SDF", "BHM", "HSV", "MOB",
    "PNS", "TLH", "JAX", "GNV", "SRQ", "PBI", "FMY", "DAB", "MLB", "TYS",
    "GSP", "CHS", "CAE", "MYR", "ILM", "AVL", "GSO", "RDU", "FAY", "OAJ",
    "SAV", "AGS", "CHA", "TRI", "ROA", "CHO", "LYH", "PHF", "HPN", "ISP",
    "EWN", "ORH",
    
    # === NORTH AMERICA - CANADA ===
    "YYZ", "YVR", "YUL", "YYC", "YEG", "YOW", "YWG", "YHZ", "YQB", "YXE",
    "YYJ", "YQR", "YXU", "YKF", "YYT", "YQM", "YFC", "YSB", "YXS", "YLW",
    
    # === NORTH AMERICA - MEXICO ===
    "MEX", "CUN", "GDL", "MTY", "TIJ", "SJD", "PVR", "MZT", "ACA", "ZIH",
    "HMO", "VER", "MID", "CZM", "OAX", "BJX", "AGU", "QRO", "SLP", "ZCL",
    
    # === CENTRAL AMERICA & CARIBBEAN ===
    "SJU", "HAV", "SDQ", "PUJ", "KIN", "MBJ", "NAS", "FPO", "GCM", "AUA",
    "CUR", "SXM", "POS", "BGI", "PTY", "SJO", "LIR", "SAL", "GUA", "TGU",
    "MGA", "RTB", "SAP", "BZE",
    
    # === SOUTH AMERICA ===
    "GRU", "GIG", "BSB", "CNF", "CWB", "POA", "REC", "SSA", "FOR", "MAO",
    "EZE", "AEP", "COR", "MDZ", "SCL", "BOG", "MDE", "CLO", "CTG", "BAQ",
    "LIM", "CUZ", "AQP", "CCS", "MAR", "VLN", "UIO", "GYE", "LPB", "VVI",
    "MVD", "ASU", "GEO", "PBM", "CAY",
    
    # === EUROPE ===
    "LHR", "CDG", "AMS", "FRA", "MAD", "BCN", "FCO", "MUC", "ZRH", "VIE",
    "CPH", "OSL", "ARN", "HEL", "DUB", "EDI", "GLA", "MAN", "BHX", "LGW",
    "STN", "LTN", "BRS", "NCL", "LPL", "EMA", "SOU", "CWL", "BFS", "ORY",
    "LYS", "MRS", "NCE", "TLS", "BOD", "NTE", "BRU", "CRL", "LIS", "OPO",
    "FAO", "AGP", "PMI", "IBZ", "VLC", "SVQ", "BIO", "MXP", "LIN", "VCE",
    "NAP", "BLQ", "FLR", "PSA", "CIA", "PMO", "CTA", "ATH", "SKG", "HER",
    "PRG", "WAW", "KRK", "WRO", "KTW", "GDN", "POZ", "BUD", "OTP", "SOF",
    "BEG", "ZAG", "LJU", "SJJ", "SKP", "TIA", "IST", "SAW", "AYT", "ESB",
    "ADB", "DLM", "BJV", "TXL", "SXF", "HAM", "DUS", "CGN", "STR", "HAJ",
    "NUE", "LEJ", "DRS", "BRE", "BSL", "GVA", "LUX", "EIN", "RTM", "BLL",
    "AAL", "GOT", "MMX", "BGO", "TRD", "SVG", "TOS", "KEF", "FAE", "TLL",
    "RIX", "VNO", "LED", "DME", "SVO", "VKO", "KJA", "CEK", "KGD", "GOJ",
    
    # === MIDDLE EAST ===
    "DXB", "AUH", "DOH", "KWI", "BAH", "MCT", "AMM", "BEY", "TLV", "CAI",
    "HRG", "SSH", "JED", "RUH", "DMM", "AHB", "TUU", "MED", "TIF",
    
    # === AFRICA ===
    "JNB", "CPT", "DUR", "NBO", "MBA", "ADD", "ACC", "LOS", "ABV", "PHC",
    "ABJ", "DKR", "DSS", "CMN", "RAK", "TUN", "ALG", "CAI", "LXR", "ASW",
    "HBE", "MPM", "LAD", "WDH", "GBE", "BUQ", "DLA", "NSI", "LBV", "BJL",
    "ROB", "NIM", "OUA", "BKO", "COO", "TIP", "MRU", "RUN", "TNR", "SEZ",
    
    # === ASIA ===
    "PEK", "PVG", "CAN", "HKG", "CTU", "SHA", "SZX", "XIY", "CKG", "KMG",
    "WUH", "NKG", "TAO", "XMN", "FOC", "CGO", "CSX", "HAK", "URC", "HRB",
    "DLC", "SHE", "TNA", "HGH", "NNG", "KWE", "LHW", "NRT", "HND", "KIX",
    "NGO", "FUK", "CTS", "OKA", "ICN", "GMP", "PUS", "CJU", "TPE", "TSA",
    "KHH", "RMQ", "BKK", "DMK", "HKT", "CNX", "HDY", "UTP", "KBV", "SGN",
    "HAN", "DAD", "CXR", "PQC", "UIH", "KUL", "PEN", "LGK", "BKI", "KCH",
    "SIN", "CGK", "SUB", "DPS", "MNL", "CEB", "DVO", "CRK", "ILO", "BCD",
    "RGN", "MDL", "PNH", "REP", "VTE", "LPQ", "BKI", "BWN", "DIL", "DEL",
    "BOM", "BLR", "MAA", "CCU", "HYD", "COK", "AMD", "GOI", "TRV", "CJB",
    "PNQ", "GAU", "BBI", "CCJ", "IXB", "CMB", "DAC", "CGP", "KTM", "ISB",
    "KHI", "LHE", "PEW", "SKT", "MUX", "FSD", "GIL", "KDU", "DYU", "KBL",
    "GYD", "EVN", "TBS", "ALA", "TSE", "TAS", "DYU", "FRU", "OSS", "ASB",
    "MCT", "SLL", "AAN", "SUV", "NAN", "APW", "HIR", "POM", "LAE", "RAR",
    
    # === OCEANIA ===
    "SYD", "MEL", "BNE", "PER", "ADL", "OOL", "CNS", "DRW", "HBA", "CBR",
    "AKL", "WLG", "CHC", "ZQN", "DUD", "NSN", "PPT", "NOU", "VLI", "WLS",
    
    # === CARGO/SPECIAL HUBS ===
    "ANC", "MEM", "SDF", "CVG", "IND", "ONT", "OAK", "RFD", "GSO", "DAY",
    "AFW", "MHR", "CGN", "LEJ", "EMA", "STN", "LGG", "HHN", "VIT", "OSR",
    "SHJ", "DWC", "MAO", "VCP", "BOG", "UIO", "LIM", "PTY", "GUA", "SJO",
}

# =================== AIRCRAFT TYPES ===================
# Format: ICAO Code: Full Name (IATA Code)

AIRCRAFT_TYPES = {
    # === AIRBUS ===
    "A124": "Antonov An-124 Ruslan (A4F)",
    "A140": "Antonov An-140 (A40)",
    "A148": "Antonov An-148 (A81)",
    "A158": "Antonov An-158 (A58)",
    "A225": "Antonov An-225 Mriya (A5F)",
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
    "A3ST": "Airbus A300-600ST Beluga (ABB)",
    "A400": "Airbus A400M Atlas",
    
    # === BOEING ===
    "B37M": "Boeing 737 MAX 7 (7M7)",
    "B38M": "Boeing 737 MAX 8 (7M8)",
    "B39M": "Boeing 737 MAX 9 (7M9)",
    "B3XM": "Boeing 737 MAX 10 (7MJ)",
    "B461": "BAe 146-100 (141)",
    "B462": "BAe 146-200 (142)",
    "B463": "BAe 146-300 (143)",
    "B52": "Boeing B-52 Stratofortress",
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
    
    # === BOMBARDIER ===
    "BCS1": "Bombardier CS100 / A220-100 (221)",
    "BCS3": "Bombardier CS300 / A220-300 (223)",
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
    
    # === ANTONOV ===
    "AN12": "Antonov An-12 (ANF)",
    "AN24": "Antonov An-24 (AN4)",
    "AN26": "Antonov An-26 (A26)",
    "AN28": "Antonov An-28 (A28)",
    "AN30": "Antonov An-30 (A30)",
    "AN32": "Antonov An-32 (A32)",
    "AN72": "Antonov An-72/74 (AN7)",
    
    # === ATR ===
    "AT43": "ATR 42-300/320 (AT4)",
    "AT45": "ATR 42-500 (AT5)",
    "AT46": "ATR 42-600 (ATR)",
    "AT72": "ATR 72-201/202 (AT7)",
    "AT73": "ATR 72-211/212 (ATR)",
    "AT75": "ATR 72-212A (500) (ATR)",
    "AT76": "ATR 72-212A (600) (ATR)",
    
    # === CESSNA ===
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
    
    # === DOUGLAS/MCDONNELL DOUGLAS ===
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
    
    # === DE HAVILLAND ===
    "DH8A": "De Havilland Canada DHC-8-100 (DH1)",
    "DH8B": "De Havilland Canada DHC-8-200 (DH2)",
    "DH8C": "De Havilland Canada DHC-8-300 (DH3)",
    "DH8D": "De Havilland Canada DHC-8-400 (DH4)",
    "DHC6": "De Havilland Canada DHC-6 Twin Otter (DHT)",
    "DHC7": "De Havilland Canada DHC-7 Dash 7 (DH7)",
    
    # === FOKKER ===
    "F100": "Fokker 100 (100)",
    "F27": "Fokker F27 Friendship (F27)",
    "F28": "Fokker F28 Fellowship (F21)",
    "F50": "Fokker 50 (F50)",
    "F70": "Fokker 70 (F70)",
    
    # === OTHERS ===
    "A748": "Hawker Siddeley HS 748 (HS7)",
    "ATP": "British Aerospace ATP (ATP)",
    "BA11": "British Aerospace BAC One-Eleven (B11)",
    "C130": "Lockheed C-130 Hercules (LOH)",
    "C5M": "Lockheed C-5M Super Galaxy",
    "C919": "Comac C919 (919)",
    "IL18": "Ilyushin Il-18 (IL8)",
    "IL62": "Ilyushin Il-62 (IL6)",
    "IL76": "Ilyushin Il-76 (IL7)",
    "IL86": "Ilyushin Il-86 (ILW)",
    "IL96": "Ilyushin Il-96 (I93)",
    "JS31": "British Aerospace Jetstream 31 (J31)",
    "JS32": "British Aerospace Jetstream 32 (J32)",
    "JS41": "British Aerospace Jetstream 41 (J41)",
    "L101": "Lockheed L-1011 TriStar (L10)",
    "L188": "Lockheed L-188 Electra (LOE)",
    "L410": "LET 410 (L4T)",
    "RJ1H": "Avro RJ100 (AR1)",
    "RJ70": "Avro RJ70 (AR7)",
    "RJ85": "Avro RJ85 (AR8)",
    "SB20": "Saab 2000 (S20)",
    "SF34": "Saab 340 (SF3)",
    "SU95": "Sukhoi Superjet 100-95 (SU9)",
    "T154": "Tupolev Tu-154 (TU5)",
    "T204": "Tupolev Tu-204/214 (T20)",
    "Y12": "Harbin Y-12 (YN2)",
    "YK40": "Yakovlev Yak-40 (YK4)",
    "YK42": "Yakovlev Yak-42 (YK2)",
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



# ---------- Enhanced Schema with Relationships ----------------
class FlightConnection(BaseModel):
    """Represents connections between flights"""

    from_flight: str
    to_flight: str
    connection_time: int  # minutes
    connection_type: str  # 'same_day', 'overnight', 'crew_change'


class Flight(BaseModel):
    date: str = Field(..., description="Flight date in MM/DD/YYYY format")
    flight_no: str = Field(..., description="Flight number including airline prefix")
    origin: Optional[str] = Field(None, description="Origin airport code")
    dest: Optional[str] = Field(None, description="Destination airport code")
    sched_out_local: Optional[str] = Field(
        None, description="Scheduled departure time in HHMM format"
    )
    sched_in_local: Optional[str] = Field(
        None, description="Scheduled arrival time in HHMM format"
    )
    duty: Optional[str] = Field(None, description="Duty time or flight duration")
    airline: Optional[str] = Field(None, description="Airline name or code")
    confidence: float = Field(0.95, description="Extraction confidence score")

    # Enhanced fields
    equipment: Optional[str] = Field(None, description="Aircraft type")
    crew_position: Optional[str] = Field(
        None, description="Crew position (CA, FO, etc)"
    )
    trip_id: Optional[str] = Field(None, description="Trip or pairing ID")
    deadhead: Optional[bool] = Field(
        None, description="Whether this is a deadhead flight"
    )

    # Relationship fields
    connection_from: Optional[str] = Field(
        None, description="Previous connected flight"
    )
    connection_to: Optional[str] = Field(None, description="Next connected flight")
    day_in_sequence: Optional[int] = Field(
        None, description="Day number in multi-day trip"
    )

    # Extraction metadata
    extraction_method: Optional[str] = Field(
        None, description="Which model extracted this"
    )
    extraction_confidence_breakdown: Optional[Dict[str, float]] = None

    @validator("flight_no")
    def clean_flight_no(cls, v):
        if not v:
            return v
        # Enhanced cleaning with pattern preservation
        cleaned = re.sub(r"[^\w\d\-/]", "", v.upper())

        # UPS-specific patterns
        if re.match(r"^[A-Z]\d{5}[A-Z]?$", cleaned):
            return cleaned
        # Standard airline format
        if re.match(r"^[A-Z]{2}\d{1,4}[A-Z]?$", cleaned):
            return cleaned
        # Numeric only (regional/codeshare)
        if re.match(r"^\d{3,5}$", cleaned):
            return cleaned
        return cleaned

    @validator("origin", "dest")
    def validate_airport(cls, v):
        if v and len(v) >= 3:
            # Try to match against known airports
            v_upper = v.upper()
            if v_upper in MAJOR_AIRPORTS:
                return v_upper
            # Check if it starts with K (US airport)
            if v_upper.startswith("K") and len(v_upper) == 4:
                return v_upper
        return v


class Result(BaseModel):
    flights: List[Flight]
    connections: List[FlightConnection] = []
    processing_time_ms: int
    total_flights_found: int
    avg_conf: float
    extraction_method: str = "ensemble"
    quality_score: float
    schedule_metadata: Optional[Dict] = None
    validation_warnings: List[str] = []


# ---------- Layout Detection and Classification ---------------
class LayoutAnalyzer:
    """Advanced layout analysis for schedule type detection"""

    @staticmethod
    def detect_schedule_characteristics(img: np.ndarray) -> Dict[str, Any]:
        """Comprehensive schedule analysis"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # Detect structural elements
        edges = cv2.Canny(gray, 50, 150)

        # Detect lines using HoughLinesP with better parameters
        horizontal_lines = []
        vertical_lines = []
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, 50, minLineLength=50, maxLineGap=20
        )

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                if angle < 10 or angle > 170:
                    horizontal_lines.append(line[0])
                elif 80 < angle < 100:
                    vertical_lines.append(line[0])

        # Detect text regions
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Analyze text distribution
        text_regions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 50:  # Filter small noise
                x, y, w_c, h_c = cv2.boundingRect(contour)
                text_regions.append((x, y, w_c, h_c))

        # Detect grid patterns (calendar detection)
        grid_score = LayoutAnalyzer._calculate_grid_score(
            horizontal_lines, vertical_lines, w, h
        )

        # Detect table patterns
        table_score = LayoutAnalyzer._calculate_table_score(
            horizontal_lines, text_regions
        )

        # Detect mobile UI elements
        mobile_score = LayoutAnalyzer._detect_mobile_ui(img)

        # Determine primary layout type
        if grid_score > 0.7 and len(horizontal_lines) > 5 and len(vertical_lines) > 5:
            layout_type = "calendar"
        elif table_score > 0.6 and len(horizontal_lines) > 3:
            layout_type = "table"
        elif mobile_score > 0.5:
            layout_type = "mobile"
        else:
            layout_type = "mixed"

        return {
            "type": layout_type,
            "grid_score": grid_score,
            "table_score": table_score,
            "mobile_score": mobile_score,
            "horizontal_lines": len(horizontal_lines),
            "vertical_lines": len(vertical_lines),
            "text_regions": len(text_regions),
            "dimensions": (h, w),
            "detected_cells": LayoutAnalyzer._detect_cells(
                horizontal_lines, vertical_lines
            )
            if layout_type == "calendar"
            else [],
        }

    @staticmethod
    def _calculate_grid_score(
        h_lines: List, v_lines: List, width: int, height: int
    ) -> float:
        """Calculate how grid-like the layout is"""
        if len(h_lines) < 3 or len(v_lines) < 3:
            return 0.0

        # Check spacing consistency
        h_positions = sorted([line[1] for line in h_lines])
        v_positions = sorted([line[0] for line in v_lines])

        h_spacings = [
            h_positions[i + 1] - h_positions[i] for i in range(len(h_positions) - 1)
        ]
        v_spacings = [
            v_positions[i + 1] - v_positions[i] for i in range(len(v_positions) - 1)
        ]

        # Calculate standard deviation
        h_std = np.std(h_spacings) if h_spacings else float("inf")
        v_std = np.std(v_spacings) if v_spacings else float("inf")

        # Lower std means more regular grid
        regularity_score = 1.0 / (1.0 + (h_std + v_std) / 100.0)

        # Check coverage
        coverage_score = min(1.0, (len(h_lines) * len(v_lines)) / 100.0)

        return 0.6 * regularity_score + 0.4 * coverage_score

    @staticmethod
    def _calculate_table_score(h_lines: List, text_regions: List) -> float:
        """Calculate how table-like the layout is"""
        if len(h_lines) < 2:
            return 0.0

        # Check if text aligns with horizontal lines
        aligned_regions = 0
        for region in text_regions:
            x, y, w, h = region
            for line in h_lines:
                if abs(y - line[1]) < 20 or abs(y + h - line[1]) < 20:
                    aligned_regions += 1
                    break

        alignment_score = aligned_regions / max(1, len(text_regions))
        row_score = min(1.0, len(h_lines) / 10.0)

        return 0.7 * alignment_score + 0.3 * row_score

    @staticmethod
    def _detect_mobile_ui(img: np.ndarray) -> float:
        """Detect mobile UI elements"""
        h, w, _ = img.shape
        aspect_ratio = h / w

        # Mobile typically has aspect ratio > 1.5
        mobile_aspect = 1.0 if aspect_ratio > 1.5 else 0.5

        # Check for status bar (top dark region)
        top_region = img[: int(h * 0.1), :]
        top_darkness = np.mean(top_region) < 100

        # Check for navigation bar (bottom region)
        bottom_region = img[int(h * 0.9) :, :]
        bottom_darkness = np.mean(bottom_region) < 100

        ui_score = 0.5 * mobile_aspect + 0.25 * top_darkness + 0.25 * bottom_darkness
        return ui_score

    @staticmethod
    def _detect_cells(h_lines: List, v_lines: List) -> List[Tuple[int, int, int, int]]:
        """Detect individual cells in a grid"""
        cells = []
        h_positions = sorted(list(set([line[1] for line in h_lines])))
        v_positions = sorted(list(set([line[0] for line in v_lines])))

        for i in range(len(h_positions) - 1):
            for j in range(len(v_positions) - 1):
                x1, y1 = v_positions[j], h_positions[i]
                x2, y2 = v_positions[j + 1], h_positions[i + 1]
                cells.append((x1, y1, x2 - x1, y2 - y1))

        return cells


# ---------- Multi-Engine OCR Processor ------------------------
class MultiEngineOCR:
    """Advanced OCR using multiple engines with voting"""

    def __init__(self):
        self.engines = ["tesseract"]
        if OCR_AVAILABLE and easy_reader is not None:
            self.engines.append("easyocr")


    async def extract_with_voting(
        self, img: np.ndarray, region: Optional[Tuple[int, int, int, int]] = None
    ) -> Dict[str, Any]:
        """Extract text using multiple OCR engines and vote on results"""
        if not self.engines:
            return {"text": "", "confidence": 0, "method": "none"}

        # Crop to region if specified
        if region:
            x, y, w, h = region
            img = img[y : y + h, x : x + w]

        results = await asyncio.gather(
            self._tesseract_ocr(img), self._easyocr_ocr(img), return_exceptions=True
        )

        # Filter out exceptions
        valid_results = [
            r for r in results if not isinstance(r, Exception) and r.get("text")
        ]

        if not valid_results:
            return {"text": "", "confidence": 0, "method": "none"}

        # Vote on best result using edit distance
        best_result = self._vote_best_result(valid_results)
        return best_result

    async def _tesseract_ocr(self, img: np.ndarray) -> Dict[str, Any]:
        """Tesseract OCR with multiple configs"""
        try:
            # Convert to PIL
            pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

            # Try multiple configs
            configs = [
                "--psm 6 -c tessedit_char_whitelist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz:/-., '",
                "--psm 11",
                "--psm 3",
            ]

            best_text = ""
            best_conf = 0

            for config in configs:
                try:
                    data = pytesseract.image_to_data(
                        pil_img, config=config, output_type=pytesseract.Output.DICT
                    )

                    # Extract text with confidence
                    words = []
                    confidences = []
                    for i, conf in enumerate(data["conf"]):
                        if int(conf) > 30:
                            words.append(data["text"][i])
                            confidences.append(int(conf))

                    text = " ".join(words).strip()
                    avg_conf = np.mean(confidences) if confidences else 0

                    if avg_conf > best_conf:
                        best_text = text
                        best_conf = avg_conf

                except:
                    continue

            return {
                "text": best_text,
                "confidence": best_conf / 100.0,
                "method": "tesseract",
            }

        except Exception as e:
            logger.error(f"Tesseract error: {e}")
            return {"text": "", "confidence": 0, "method": "tesseract"}

    async def _easyocr_ocr(self, img: np.ndarray) -> Dict[str, Any]:
        """EasyOCR extraction"""
        try:
            if easy_reader is None:
                return {"text": "", "confidence": 0, "method": "easyocr"}

            results = easy_reader.readtext(img, detail=1, paragraph=False)

            if not results:
                return {"text": "", "confidence": 0, "method": "easyocr"}

            # Sort by position and combine
            sorted_results = sorted(results, key=lambda x: (x[0][0][1], x[0][0][0]))

            texts = []
            confidences = []
            for bbox, text, conf in sorted_results:
                if conf > 0.3:
                    texts.append(text)
                    confidences.append(conf)

            combined_text = " ".join(texts)
            avg_conf = np.mean(confidences) if confidences else 0

            return {"text": combined_text, "confidence": avg_conf, "method": "easyocr"}

        except Exception as e:
            logger.error(f"EasyOCR error: {e}")
            return {"text": "", "confidence": 0, "method": "easyocr"}

    def _vote_best_result(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Vote on best OCR result using edit distance"""
        if len(results) == 1:
            return results[0]

        # Calculate pairwise edit distances
        scores = []
        for i, result in enumerate(results):
            distances = []
            for j, other in enumerate(results):
                if i != j:
                    dist = Levenshtein.distance(result["text"], other["text"])
                    # Normalize by length
                    norm_dist = dist / max(
                        1, max(len(result["text"]), len(other["text"]))
                    )
                    distances.append(1 - norm_dist)

            # Weight by confidence
            avg_similarity = np.mean(distances) if distances else 0
            weighted_score = avg_similarity * result["confidence"]
            scores.append(weighted_score)

        # Return result with highest score
        best_idx = np.argmax(scores)
        return results[best_idx]


# ---------- Enhanced Image Processor --------------------------
class EnhancedImageProcessor:
    """Advanced image processing with quality assessment"""

    @staticmethod
    def assess_image_quality(img: np.ndarray) -> Dict[str, float]:
        """Assess image quality metrics"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Blur detection using Laplacian variance
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        blur_score = min(1.0, laplacian_var / 1000.0)

        # Contrast assessment
        contrast = gray.std()
        contrast_score = min(1.0, contrast / 50.0)

        # Noise estimation
        noise = EnhancedImageProcessor._estimate_noise(gray)
        noise_score = max(0, 1.0 - noise / 50.0)

        # Text density
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        text_pixels = np.sum(binary == 0)
        text_density = text_pixels / (gray.shape[0] * gray.shape[1])

        # Overall quality score
        quality_score = (
            0.3 * blur_score
            + 0.3 * contrast_score
            + 0.2 * noise_score
            + 0.2 * min(1.0, text_density * 10)
        )

        return {
            "overall": quality_score,
            "blur": blur_score,
            "contrast": contrast_score,
            "noise": noise_score,
            "text_density": text_density,
        }

    @staticmethod
    def _estimate_noise(gray: np.ndarray) -> float:
        """Estimate image noise level"""
        # Use median absolute deviation
        h, w = gray.shape
        gray_float = gray.astype(np.float64)

        # Calculate local variations
        dx = gray_float[1:, :] - gray_float[:-1, :]
        dy = gray_float[:, 1:] - gray_float[:, :-1]

        # Median absolute deviation
        mad_x = np.median(np.abs(dx - np.median(dx)))
        mad_y = np.median(np.abs(dy - np.median(dy)))

        noise_estimate = (mad_x + mad_y) / 2.0
        return noise_estimate

    @staticmethod
    def adaptive_enhancement(
        img: np.ndarray, quality_metrics: Dict[str, float]
    ) -> List[np.ndarray]:
        """Create enhanced versions based on quality assessment"""
        enhanced_versions = [img]  # Always include original

        # If blurry, apply sharpening
        if quality_metrics["blur"] < 0.5:
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            sharpened = cv2.filter2D(img, -1, kernel)
            enhanced_versions.append(sharpened)

        # If low contrast, apply CLAHE
        if quality_metrics["contrast"] < 0.5:
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            enhanced = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
            enhanced_versions.append(enhanced)

        # If noisy, apply denoising
        if quality_metrics["noise"] < 0.7:
            denoised = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
            enhanced_versions.append(denoised)

        # Always create a high-contrast version for OCR
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        enhanced_versions.append(cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR))

        return enhanced_versions

    @staticmethod
    def extract_roi_for_schedules(
        img: np.ndarray,
        layout_info: Dict
    ) -> List[Tuple[np.ndarray, str]]:
        """Extract regions of interest based on layout type (with header first)."""
        rois: List[Tuple[np.ndarray, str]] = []
        h, w = img.shape[:2]

        # ─── Always grab the top ~15% as "header" so we pick up base/crew info ───
        header_h = int(h * 0.15)
        rois.append((img[:header_h, :], "header"))

        if layout_info["type"] == "calendar" and layout_info.get("detected_cells"):
            # Extract each detected cell (calendar‑style)
            for i, (x, y, cw, ch) in enumerate(layout_info["detected_cells"][:50]):
                if cw > 30 and ch > 30:  # skip tiny noise
                    cell_img = img[y : y + ch, x : x + cw]
                    rois.append((cell_img, f"cell_{i}"))

        elif layout_info["type"] == "table":
            # Slice into horizontal rows
            row_height = h // max(3, layout_info["horizontal_lines"])
            for i in range(min(20, layout_info["horizontal_lines"] - 1)):
                y0 = i * row_height
                y1 = (i + 1) * row_height
                row_img = img[y0:y1, :]
                rois.append((row_img, f"row_{i}"))

        else:
            # Mixed/mobile UI fallback: header + 3 equal chunks
            # (header already added; now add 3 middle sections)
            for i in range(3):
                y0 = int(h * (0.15 + i * 0.28))
                y1 = int(h * (0.15 + (i + 1) * 0.28))
                section_img = img[y0:y1, :]
                rois.append((section_img, f"section_{i}"))

        return rois


def _safe_parse_date(date_str: str) -> Optional[datetime]:
    """Handle both 05/01/2025 and 2025-05-01 (and fall back to dateutil)."""
    for fmt in ("%m/%d/%Y", "%Y-%m-%d", "%m/%d/%y"):
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    try:
        return parser.parse(date_str, fuzzy=True)
    except Exception:
        return None


# ---------- Context-Aware Flight Extractor --------------------
class FlightRelationshipExtractor:
    """Extract relationships between flights"""

    @staticmethod
    def detect_connections(flights: List[Flight]) -> List[FlightConnection]:
        """Detect connections between flights"""
        connections = []

        # Sort flights by date and time
        sorted_flights = sorted(
            flights, key=lambda f: (f.date, f.sched_out_local or "0000")
        )

        for i in range(len(sorted_flights) - 1):
            current = sorted_flights[i]
            next_flight = sorted_flights[i + 1]

            # Check if same day
            if current.date == next_flight.date:
                # Check if arrival airport matches departure
                if (
                    current.dest
                    and next_flight.origin
                    and current.dest == next_flight.origin
                ):
                    # Calculate connection time
                    if current.sched_in_local and next_flight.sched_out_local:
                        arr_time = FlightRelationshipExtractor._parse_time(
                            current.sched_in_local
                        )
                        dep_time = FlightRelationshipExtractor._parse_time(
                            next_flight.sched_out_local
                        )

                        if arr_time and dep_time:
                            conn_time = (dep_time - arr_time).total_seconds() / 60

                            if 30 <= conn_time <= 300:  # 30 min to 5 hours
                                connections.append(
                                    FlightConnection(
                                        from_flight=current.flight_no,
                                        to_flight=next_flight.flight_no,
                                        connection_time=int(conn_time),
                                        connection_type="same_day",
                                    )
                                )

            # Check for overnight connections
            else:
                current_date = _safe_parse_date(current.date)
                next_date = _safe_parse_date(next_flight.date)
                if not current_date or not next_date:
                    continue
                if (next_date - current_date).days == 1:
                    if (
                        current.dest
                        and next_flight.origin
                        and current.dest == next_flight.origin
                    ):
                        connections.append(
                            FlightConnection(
                                from_flight=current.flight_no,
                                to_flight=next_flight.flight_no,
                                connection_time=0,  # Unknown
                                connection_type="overnight",
                            )
                        )

        return connections

    @staticmethod
    def detect_multi_day_trips(flights: List[Flight]) -> Dict[str, List[Flight]]:
        """Group flights into multi-day trips"""
        trips = defaultdict(list)

        # Group by pairing/trip ID if available
        for flight in flights:
            if flight.trip_id:
                trips[flight.trip_id].append(flight)

        # Also try to detect based on patterns
        sorted_flights = sorted(
            flights, key=lambda f: (f.date, f.sched_out_local or "0000")
        )

        current_trip = []
        trip_counter = 1

        for i, flight in enumerate(sorted_flights):
            if not current_trip:
                current_trip.append(flight)
            else:
                # Check if this could be part of same trip
                last_flight = current_trip[-1]

                # Same day continuation
                if flight.date == last_flight.date:
                    current_trip.append(flight)
                else:
                    # Check if overnight at same location
                    flight_date = _safe_parse_date(flight.date)
                    last_date = _safe_parse_date(last_flight.date)
                    if not flight_date or not last_date:
                        continue

                    if (
                        flight_date - last_date
                    ).days == 1 and last_flight.dest == flight.origin:
                        current_trip.append(flight)
                    else:
                        # New trip
                        if len(current_trip) > 1:
                            trips[f"trip_{trip_counter}"] = current_trip.copy()
                            trip_counter += 1
                        current_trip = [flight]

        # Don't forget last trip
        if len(current_trip) > 1:
            trips[f"trip_{trip_counter}"] = current_trip

        return dict(trips)

    @staticmethod
    def _parse_time(time_str: str) -> Optional[datetime]:
        """Parse time string to datetime"""
        try:
            if len(time_str) == 4:
                hour = int(time_str[:2])
                minute = int(time_str[2:])
                return datetime.now().replace(
                    hour=hour, minute=minute, second=0, microsecond=0
                )
            elif ":" in time_str:
                parts = time_str.split(":")
                hour = int(parts[0])
                minute = int(parts[1])
                return datetime.now().replace(
                    hour=hour, minute=minute, second=0, microsecond=0
                )
        except:
            return None


# ---------- Domain-Specific Validation ------------------------
class AviationValidator:
    """Aviation-specific validation rules"""

    @staticmethod
    def validate_flight(flight: Flight) -> Tuple[bool, List[str]]:
        """Validate flight data against aviation rules"""
        warnings = []

        # Validate flight number format
        if flight.flight_no:
            if not re.match(r"^[A-Z0-9]{2,7}[A-Z]?$", flight.flight_no):
                warnings.append(f"Unusual flight number format: {flight.flight_no}")

        # Validate airports
        if flight.origin and flight.origin not in MAJOR_AIRPORTS:
            if not flight.origin.startswith("K") or len(flight.origin) != 4:
                warnings.append(f"Unknown origin airport: {flight.origin}")

        if flight.dest and flight.dest not in MAJOR_AIRPORTS:
            if not flight.dest.startswith("K") or len(flight.dest) != 4:
                warnings.append(f"Unknown destination airport: {flight.dest}")

        # Validate times
        if flight.sched_out_local and flight.sched_in_local:
            try:
                dep_hour = int(flight.sched_out_local[:2])
                dep_min = int(flight.sched_out_local[2:])
                arr_hour = int(flight.sched_in_local[:2])
                arr_min = int(flight.sched_in_local[2:])

                if not (0 <= dep_hour <= 23 and 0 <= dep_min <= 59):
                    warnings.append(f"Invalid departure time: {flight.sched_out_local}")

                if not (0 <= arr_hour <= 23 and 0 <= arr_min <= 59):
                    warnings.append(f"Invalid arrival time: {flight.sched_in_local}")

                # Check if arrival is before departure (same day)
                if flight.sched_in_local < flight.sched_out_local:
                    # Could be overnight flight - not necessarily an error
                    pass

            except:
                warnings.append("Invalid time format")

        # Validate date
        try:
            datetime.strptime(flight.date, "%m/%d/%Y")
        except:
            warnings.append(f"Invalid date format: {flight.date}")

        return len(warnings) == 0, warnings

    @staticmethod
    def validate_schedule(flights: List[Flight]) -> List[str]:
        """Validate entire schedule for consistency"""
        warnings = []

        # Check for duplicate flights
        flight_keys = [(f.date, f.flight_no or "", f.sched_out_local or "") for f in flights]
        duplicates = [k for k, count in Counter(flight_keys).items() if count > 1]

        if duplicates:
            warnings.append(f"Duplicate flights detected: {duplicates}")

        # Check for impossible connections
        for i in range(len(flights) - 1):
            current = flights[i]
            next_flight = flights[i + 1]

            if current.date == next_flight.date and current.dest and next_flight.origin:
                if current.dest != next_flight.origin:
                    # Check time gap
                    if current.sched_in_local and next_flight.sched_out_local:
                        gap_minutes = AviationValidator._calculate_time_gap(
                            current.sched_in_local, next_flight.sched_out_local
                        )

                        if gap_minutes < 120:  # Less than 2 hours for airport change
                            warnings.append(
                                f"Insufficient time for airport change: {current.dest} to {next_flight.origin}"
                            )

        return warnings

    @staticmethod
    def _calculate_time_gap(time1: str, time2: str) -> int:
        """Calculate minutes between two time strings"""
        try:
            t1 = datetime.strptime(time1, "%H%M")
            t2 = datetime.strptime(time2, "%H%M")
            return int((t2 - t1).total_seconds() / 60)
        except:
            return 999  # Large number if can't parse


# ---------- Enhanced GPT-4V Extraction ------------------------
class EnhancedGPTExtractor:
    """Advanced GPT-4V extraction with better prompting"""

    # ---------- MEGA‑ULTRA PROMPT GENERATOR --------------------
    @staticmethod
    def create_extraction_prompt(
        layout_info: Dict, ocr_data: Dict, quality_metrics: Dict
    ) -> str:
        """
        Build the final‑boss prompt that merges every previous draft,
        airline‑specific rules, layout hints & OCR intelligence.
        """

        # ───────────────────────────────────────────────
        # Helper lambdas
        # ───────────────────────────────────────────────
        j = lambda key, n: ", ".join(ocr_data.get(key, [])[:n]) or "—"
        layout_type = layout_info.get("type", "unknown")
        h_lines = layout_info.get("horizontal_lines", 0)
        v_lines = layout_info.get("vertical_lines", 0)
        quality = quality_metrics.get("overall", 0.0)

        # OCR quick‑look strings
        ocr_airlines = j("airlines", 6)
        ocr_dates = j("dates", 8)
        ocr_times = j("times", 15)
        ocr_airports = j("airports", 15)
        ocr_flight_numbers = j("flight_numbers", 15)

        # Primary airline guess
        primary_airline = ((ocr_data.get("airlines") or ["Unknown"])[0]).upper()

        # ───────────────────────────────────────────────
        # Airline‑specific rule blocks
        # ───────────────────────────────────────────────
        AIRLINE_RULES = {
            "UPS": """
=== UPS PILOT ROSTER SPECIFICS ===
🛩️ Pairing code A70186R → A=Trip type, 70186=unique, R=Crew pos  
⏰ Time cols: Sched Out • Out • Sched In • In  
🏢 Hubs & equip: SDF, CVG, LOU / B744F, B748F, MD11F, A300F
""",
            "FEDEX": """
=== FEDEX PILOT ROSTER SPECIFICS ===
🛩️ Trip #’s numeric (123 / 456 / 789)  
⏰ Times always HHMM local  
🏢 Equip: B777F, B767F, MD11F, ATR72F, C208  
🌍 Hubs: MEM, IND, OAK, CDG, CGN
""",
            "AMERICAN": """
=== AMERICAN AIRLINES SPECIFICS ===
🛩️ Flight # AA####  
⏰ 4‑day pairings common (PBS)  
🏢 Equip: A321, B737, B777, B787  
🌍 Hubs: DFW, CLT, PHX, PHL, MIA, LAX, JFK
""",
            "UNITED": """
=== UNITED AIRLINES SPECIFICS ===
🛩️ Flight # UA####  
⏰ CCS / VIPS roster screens  
🏢 Equip: B737, A320, B777, B787, B767  
🌍 Hubs: ORD, DEN, SFO, IAH, EWR, IAD, LAX
""",
            "DELTA": """
=== DELTA AIR LINES SPECIFICS ===
🛩️ Flight # DL####  
⏰ CCS schedules; trips 1‑5 days  
🏢 Equip: A320, A330, B737, B757, B767  
🌍 Hubs: ATL, MSP, DTW, SEA, SLC, BOS, JFK, LAX
""",
        }

        def detect_airline_block(airline_guess: str, raw_ocr: str) -> str:
            ag = airline_guess.upper()
            if any(k in ag for k in ["UPS", "A70186R"]):
                return AIRLINE_RULES["UPS"]
            if any(k in ag for k in ["FEDEX", "FDX"]):
                return AIRLINE_RULES["FEDEX"]
            if any(k in ag for k in ["AMERICAN", "AA"]):
                return AIRLINE_RULES["AMERICAN"]
            if any(k in ag for k in ["UNITED", "UA"]):
                return AIRLINE_RULES["UNITED"]
            if any(k in ag for k in ["DELTA", "DL"]):
                return AIRLINE_RULES["DELTA"]
            return """
=== GENERIC AIRLINE FORMAT ===
🛩️ Flight #: Standard IATA (XX####)  
⏰ Times  : HHMM local / Zulu  
🏢 Equip  : A320, B737, regional jets  
🌍 Routes : Domestic & international
"""

        airline_rules_block = detect_airline_block(
            primary_airline, " ".join(ocr_data.get("raw_text", []))
        )

        # Low‑quality nudge section
        low_quality_nudge_block = ""
        if quality < 0.50:
            low_quality_nudge_block = """
IMAGE QUALITY WARNING – low sharpness/contrast.
• Use context clues and pattern inference.  
• Where digits are partially missing, guess intelligently.  
• Still extract everything; lower confidence scores as needed.
"""

        # ───────────────────────────────────────────────
        # MEGA‑ULTRA PROMPT (verbatim)
        # ───────────────────────────────────────────────
        mega_prompt = f"""
┌──────────────────────────────────────────────────────────────────────────────┐
│  ███  FLIGHT‑INTEL OMEGA  ―  THE MEGA‑ULTRA EXTRACTION PROMPT  ███          │
└──────────────────────────────────────────────────────────────────────────────┘

You are **FLIGHT‑INTEL OMEGA**, the apex airline‑schedule analyst endowed with
the combined expertise of senior captains, crew‑scheduling leads, operations
managers and aviation data scientists.

Your supernatural abilities include:  
✈️  Perfect recognition of **all** roster formats worldwide  
🔍 Microscopic attention to detail – you **never** miss a flight  
🧠 Instant pattern recognition across layout systems & airline quirks  
⚡ Lightning‑fast extraction with surgical precision  

A missed flight = mission failure.

===============================================================================
                             ░ CONTEXT SNAPSHOT ░
===============================================================================
• Layout detected  : **{layout_type}**  
• Grid structure   : {h_lines} horizontal × {v_lines} vertical lines  
• Overall quality  : {quality:.2f} (0 = poor, 1 = perfect)  
• Primary airline  : {primary_airline}  

===============================================================================
                            ░ OCR CHEAT‑SHEET ░
===============================================================================
Airlines  → {ocr_airlines}  
Dates     → {ocr_dates}  
Times     → {ocr_times}  
Airports  → {ocr_airports}  
Flight #  → {ocr_flight_numbers}  

===============================================================================
                        ░ LAYOUT‑SPECIFIC PLAYBOOK ░
===============================================================================
▲ CALENDAR PLAYBOOK
  • Traverse cells left→right, top→bottom; each cell may hide MULTIPLE flights.  
  • Blue bar = flight leg ▍ Grey = continuation ▍ Green = reserve ▍ Yellow dot = off.  
  • Combine cell day‑number with header month to form full dates.

▲ TABLE PLAYBOOK
  • First row = headers; each subsequent row = one duty/leg.  
  • Merged/indented rows indicate connections (fill `connection_from`/`connection_to`).  
  • Ignore summary rows except for QC counts.

▲ MIXED / MOBILE PLAYBOOK
  • Disregard UI chrome/navigation.  
  • Conceptually “expand” accordion/pagination to capture every duty.

===============================================================================
                   ░ AIRLINE‑SPECIFIC EXTRACTION RULES ░
===============================================================================
{airline_rules_block}

===============================================================================
                  ░ UNIVERSAL EXTRACTION PROTOCOLS ░
===============================================================================
🎯 **FLIGHT IDENTIFIERS** – legacy, cargo, regional, charter, tail numbers.  
🗓️ **DATE FORMATS** – MM/DD/YYYY, DDMMMYY, verbal, Julian, etc.  
⏰ **TIME FORMATS** – HHMM, H:MM AM/PM, Zulu (Z), Local (L), +1 arrivals.  
🌍 **AIRPORT CODES** – IATA + ICAO, incl. military & remote strips.

===============================================================================
                ░ ADVANCED VISUAL EXTRACTION TECHNIQUES ░
===============================================================================
1 Grid scanning 2 Column mapping 3 Row clustering 4 Color/icon cues 5 QC totals  

===============================================================================
                       ░ CRITICAL FLIGHT CATEGORIES ░
===============================================================================
✓ Revenue ✓ Deadhead (DH/DHD) ✓ Ferry ✓ Training (SIM/IOE) ✓ Reserve/Standby  
✓ Codeshare ✓ Cargo‑only ✓ Charter / ad‑hoc ✓ Multi‑leg trips

===============================================================================
                         ░ CONFIDENCE MATRIX ░
===============================================================================
0.95‑1.00 = crystal clear │ 0.85‑0.94 = minor artifacts │
0.70‑0.84 = interpretation │ 0.60‑0.69 = context guess │ < 0.60 = still extract

{low_quality_nudge_block}
===============================================================================
                      ░ SELF‑CHECK BEFORE SUBMIT ░
===============================================================================
1 Count blue/grey blocks or data rows → *n* Ensure **len(flights) ≥ n**  
2 Return `null` for unknowns, KEEP original date format  
3 Provide JSON via the `extract_complete_flight_schedule` function only  

===============================================================================
                               ░ FINAL MANDATE ░
===============================================================================
Extract with the precision of a forensic investigator.  
**Missing a flight is NOT an option.**
"""

        return mega_prompt

    @staticmethod
    async def extract_with_gpt4v(
        images: List[str], layout_info: Dict, ocr_data: Dict, quality_metrics: Dict
    ) -> Tuple[List[Flight], Dict]:
        """Extract flights using GPT-4V with enhanced prompting"""

        prompt = EnhancedGPTExtractor.create_extraction_prompt(
            layout_info, ocr_data, quality_metrics
        )

        # Enhanced function schema
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "extract_complete_flight_schedule",
                    "description": "Extract all flight information from airline crew schedule",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "schedule_metadata": {
                                "type": "object",
                                "properties": {
                                    "airline": {"type": "string"},
                                    "crew_member": {"type": "string"},
                                    "base": {"type": "string"},
                                    "schedule_period": {"type": "string"},
                                    "total_flights_visible": {
                                        "type": "integer",
                                        "description": "Total number of flights you can see",
                                    },
                                },
                            },
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
                                        "equipment": {"type": "string"},
                                        "crew_position": {"type": "string"},
                                        "trip_id": {"type": "string"},
                                        "deadhead": {"type": "boolean"},
                                        "duty": {"type": "string"},
                                        "confidence": {"type": "number"},
                                        "extraction_notes": {"type": "string"},
                                    },
                                    "required": ["date", "flight_no", "confidence"],
                                },
                            },
                        },
                        "required": ["flights", "schedule_metadata"],
                    },
                },
            }
        ]

        messages = [
            {
                "role": "system",
                "content": "You are the world's best airline schedule extraction AI.",
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}]
                + [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{img}",
                            "detail": "high",
                        },
                    }
                    for img in images
                ],
            },
        ]

        try:
            response = await openai_client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=tools,
                tool_choice={
                    "type": "function",
                    "function": {"name": "extract_complete_flight_schedule"},
                },
                temperature=TEMP,
                max_tokens=MAX_TOKENS,
                top_p=0.95,
            )

            if response.choices[0].message.tool_calls:
                raw_data = json.loads(
                    response.choices[0].message.tool_calls[0].function.arguments
                )

                flights = []
                metadata = raw_data.get("schedule_metadata", {})
                

                for idx, flight_data in enumerate(raw_data.get("flights", [])):
                    if not flight_data.get("flight_no"):
                        fallback = next(
                            (
                                num
                                for num in ocr_data["flight_numbers"]
                                if num
                                not in [f.get("flight_no") for f in raw_data["flights"]]
                            ),
                            None,
                        )
                        flight_data["flight_no"] = fallback or f"UNK{idx+1:03d}"
                        flight_data["confidence"] *= 0.8
                        flight_data["extraction_notes"] = (
                            flight_data.get("extraction_notes", "")
                            + " | flight_no filled automatically"
                        )

                    if not flight_data.get("airline") and metadata.get("airline"):
                        flight_data["airline"] = metadata["airline"]

                    flight_data["extraction_method"] = "gpt-4v"
                    flights.append(Flight(**flight_data))
                    metadata["total_flights_visible"] = metadata.get("total_flights_visible", 0) + 1
                    
                return flights, metadata  # ← keep this!

        except Exception as e:
            logger.error(f"GPT-4V extraction error: {e}")

        return [], {}


# ---------- Ensemble Extraction Manager -----------------------
class EnsembleExtractor:
    """Manages multiple extraction methods and combines results"""

    def __init__(self):
        self.ocr_processor = MultiEngineOCR()
        self.image_processor = EnhancedImageProcessor()
        self.layout_analyzer = LayoutAnalyzer()
        self.gpt_extractor = EnhancedGPTExtractor()
        self.relationship_extractor = FlightRelationshipExtractor()
        self.validator = AviationValidator()

    async def extract_flights(self, img: np.ndarray) -> Result:
        """Main extraction pipeline using ensemble methods"""
        start_time = datetime.now()

        # Step 1: Analyze layout
        layout_info = self.layout_analyzer.detect_schedule_characteristics(img)
        logger.info(f"Layout detected: {layout_info['type']}")

        # Step 2: Assess quality
        quality_metrics = self.image_processor.assess_image_quality(img)
        logger.info(f"Image quality: {quality_metrics['overall']:.2f}")

        # Step 3: Adaptive enhancement
        enhanced_images = self.image_processor.adaptive_enhancement(
            img, quality_metrics
        )

        # Step 4: Extract ROIs
        rois = self.image_processor.extract_roi_for_schedules(img, layout_info)

        # Step 5: Multi-engine OCR
        ocr_results = []
        for roi_img, roi_name in rois[:10]:  # Limit ROIs
            result = await self.ocr_processor.extract_with_voting(roi_img)
            ocr_results.append((roi_name, result))

        # Step 6: Aggregate OCR data
        ocr_data = self._aggregate_ocr_data(ocr_results)

        # Step 7: Prepare images for GPT-4V
        b64_images = []
        for enhanced_img in enhanced_images[:3]:  # Limit to 3 best versions
            _, buffer = cv2.imencode(".png", enhanced_img)
            b64 = base64.b64encode(buffer).decode("utf-8")
            b64_images.append(b64)

        # Step 8: GPT-4V extraction
        gpt_flights, schedule_metadata = await self.gpt_extractor.extract_with_gpt4v(
            b64_images, layout_info, ocr_data, quality_metrics
        )
        self._inject_airports_and_duty(gpt_flights, ocr_data["raw_text"])
        self._inject_airports_from_summary(gpt_flights, ocr_data["raw_text"])

        gpt_flights = self._inherit_missing_flightnos(gpt_flights)
        for f in gpt_flights:
            f.date = self._normalise_date(f.date)

        # Step 9: Validate and correct
        validated_flights = []
        all_warnings = []

        for flight in gpt_flights:
            is_valid, warnings = self.validator.validate_flight(flight)
            if warnings:
                all_warnings.extend(warnings)

            # Attempt correction for common issues
            flight = self._correct_common_errors(flight, ocr_data)
            validated_flights.append(flight)

        # Step 10: Detect relationships
        connections = self.relationship_extractor.detect_connections(validated_flights)
        trips = self.relationship_extractor.detect_multi_day_trips(validated_flights)

        # Step 11: Add relationship data to flights
        for flight in validated_flights:
            # Mark connections
            for conn in connections:
                if flight.flight_no == conn.from_flight:
                    flight.connection_to = conn.to_flight
                elif flight.flight_no == conn.to_flight:
                    flight.connection_from = conn.from_flight

            # Mark trip associations
            for trip_id, trip_flights in trips.items():
                if any(
                    f.flight_no == flight.flight_no and f.date == flight.date
                    for f in trip_flights
                ):
                    flight.trip_id = trip_id

        # Step 12: Final validation
        schedule_warnings = self.validator.validate_schedule(validated_flights)
        all_warnings.extend(schedule_warnings)

        # Calculate metrics
        processing_time = int((datetime.now() - start_time).total_seconds() * 1000)
        avg_confidence = (
            sum(f.confidence for f in validated_flights) / len(validated_flights)
            if validated_flights
            else 0
        )

        # Quality score calculation
        quality_score = self._calculate_quality_score(
            validated_flights, ocr_data, layout_info, quality_metrics, all_warnings
        )

        schedule_metadata["total_flights_visible"] = len(validated_flights)
        default_airline = schedule_metadata.get("airline")
        for f in validated_flights:
             if not f.airline and default_airline:
                 f.airline = default_airline


        return Result(
            flights=validated_flights,
            connections=connections,
            processing_time_ms=processing_time,
            total_flights_found=len(validated_flights),
            avg_conf=avg_confidence,
            quality_score=quality_score,
            schedule_metadata=schedule_metadata,
            validation_warnings=all_warnings[:10],  # Limit warnings
        )
    
    def _inject_airports_from_summary(self, flights: List[Flight], raw_text: str):
        # same body as above, but use `self` instead of top‑level
        codes_pattern = "|".join(MAJOR_AIRPORTS)
        for f in flights:
            if not f.origin:
                m = re.search(rf"{re.escape(f.flight_no)}\D+({codes_pattern})", raw_text)
                if m:
                    f.origin = m.group(1)
            if f.origin and not f.dest:
                m = re.search(rf"{re.escape(f.flight_no)}.*?{f.origin}\D+([A-Z]{{3}})", raw_text)
                if m:
                    f.dest = m.group(1)


    def _aggregate_ocr_data(self, ocr_results: List[Tuple[str, Dict]]) -> Dict:
        """Aggregate OCR results from multiple regions"""
        aggregated = {
            "dates": set(),
            "times": set(),
            "airports": set(),
            "flight_numbers": set(),
            "airlines": set(),  # ← NEW bucket
            "raw_text": [],
        }

        # ─── Regex helpers (ADD THESE) ──────────────────────────────
        airline_code_re = re.compile(
            r"\b(UPS|FDX|AA|UA|DL|WN|B6|AS|NK|F9|AC|BA)\b", re.I
        )
        airline_name_re = re.compile(
            r"\b(UPS|FE?DEX|AMERICAN|UNITED|DELTA|SOUTHWEST|JETBLUE|ALASKA|"
            r"SPIRIT|FRONTIER|AIR\s*CANADA|BRITISH)\b",
            re.I,
        )
        # ────────────────────────────────────────────────────────────

        # Existing patterns (unchanged)
        date_pattern = re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b")
        time_pattern = re.compile(r"\b\d{4}\b|\b\d{1,2}:\d{2}\b")
        flight_pattern = re.compile(r"\b[A-Z]{1,2}\d{3,5}[A-Z]?\b|\b[A-Z]\d{5}[A-Z]\b")

        def _looks_like_flight_row(txt: str) -> bool:
            return (
                re.search(r"\b[A-Z]\d{3,5}\b", txt)        # flight number
                or re.search(r"\b\d{4}\b", txt)            # 4‑digit flight
                or re.search(r"\b[A-Z]{3}\s+[A-Z]{3}\b", txt)  # ORIGIN DEST
            )


        for roi_name, result in ocr_results:
            if result.get("confidence", 0) < 0.50:
                continue

            text = result.get("text", "")
            if not text:
                continue

            aggregated["raw_text"].append(f"{roi_name}: {text}")

            # NEW: airline hits -------------------------------------------------
            aggregated["airlines"].update(
                m.upper() for m in airline_code_re.findall(text)
            )
            aggregated["airlines"].update(
                m.upper().replace(" ", "") for m in airline_name_re.findall(text)
            )
            # -------------------------------------------------------------------

            # Existing extraction logic (unchanged)
            aggregated["dates"].update(date_pattern.findall(text))
            aggregated["times"].update(time_pattern.findall(text))
            aggregated["flight_numbers"].update(flight_pattern.findall(text))

            words = re.findall(r"\b[A-Z]{3,4}\b", text)
            for word in words:
                if word in MAJOR_AIRPORTS or (word.startswith("K") and len(word) == 4):
                    aggregated["airports"].add(word)

        # ---------- return dict (add airlines) ----------
        return {
            "dates": list(aggregated["dates"]),
            "times": list(aggregated["times"]),
            "airports": list(aggregated["airports"]),
            "flight_numbers": list(aggregated["flight_numbers"]),
            "airlines": list(aggregated["airlines"]),  # ← NEW
            "raw_text": " ".join(aggregated["raw_text"])[:2000],
        }

    def _inject_airports_and_duty(self, flights: List[Flight], raw: str):
        """
        Pulls duty hours and both origin/dest from the “EXP TAFB” summary line.
        """
        for f in flights:
            # look for e.g. "6542 EXP TAFB 90.58 SEA 3 DFW 2 PIT 2"
            pat = rf"{re.escape(f.flight_no)}\s+EXP TAFB\s+([\d\.]+)\s+([A-Z]{{3}}).+?([A-Z]{{3}})"
            m = re.search(pat, raw)
            if not m:
                continue
            duty_hrs, origin, dest = m.groups()
            f.duty = f"EXP TAFB {duty_hrs}"
            f.origin = origin
            f.dest = dest


    def _correct_common_errors(self, flight: Flight, ocr_data: Dict) -> Flight:
        """Correct common extraction errors"""
        for attr in ("sched_out_local", "sched_in_local"):
            t = getattr(flight, attr)
            if t:
                try:
                    ival = int(t)
                    if not (0 <= ival <= 2359):
                        setattr(flight, attr, None)
                    else:
                        # re‑zero pad (e.g.  830 → “0830”)
                        setattr(flight, attr, f"{ival:04d}")
                except:
                    setattr(flight, attr, None)
        if flight.dest == "DH":
            flight.deadhead = True
            flight.dest = None


        # Fix common OCR errors in flight numbers
        if flight.flight_no:
            # O/0 confusion
            flight.flight_no = flight.flight_no.replace("O", "0")

            # Check against OCR data
            if flight.flight_no not in ocr_data["flight_numbers"]:
                # Try fuzzy matching
                for ocr_flight in ocr_data["flight_numbers"]:
                    if Levenshtein.distance(flight.flight_no, ocr_flight) <= 1:
                        flight.flight_no = ocr_flight
                        flight.confidence *= 0.9
                        break

        # Validate and fix times
        if flight.sched_out_local and len(flight.sched_out_local) != 4:
            # Try to fix format
            time_match = re.search(r"(\d{1,2}):?(\d{2})", flight.sched_out_local)
            if time_match:
                hour, minute = time_match.groups()
                flight.sched_out_local = f"{hour.zfill(2)}{minute}"

        # Fix airport codes
        if flight.origin and len(flight.origin) == 2:
            # Might be airline code mistaken for airport
            flight.origin = None
            flight.confidence *= 0.8

        return flight

    @staticmethod
    def _normalise_date(d: str) -> str:
        # ─── handle DDMMMYY (e.g. "04JUL25") ─────────────────────────
        m = re.fullmatch(r"(\d{1,2})([A-Za-z]{3})(\d{2})", d)
        if m:
            day, mon_abbr, yy = m.groups()
            months = {"JAN":"01","FEB":"02","MAR":"03","APR":"04","MAY":"05","JUN":"06",
                    "JUL":"07","AUG":"08","SEP":"09","OCT":"10","NOV":"11","DEC":"12"}
            mm = months.get(mon_abbr.upper(), "01")
            year = 2000 + int(yy) if len(yy) == 2 else int(yy)
            return f"{mm}/{int(day):02d}/{year}"


        # ─── existing ISO and MM/DD fall‑back ─────────────────────────
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", d):
            return datetime.strptime(d, "%Y-%m-%d").strftime("%m/%d/%Y")
        if re.fullmatch(r"\d{2}/\d{2}", d):
            assumed_year = "2025"
            return f"{d}/{assumed_year}"
        if re.fullmatch(r"\d{2}/\d{2}/\d{2}", d):
            mm, dd, yy = d.split("/")
            return f"{mm}/{dd}/20{yy}"
        return d

    @staticmethod
    def _inherit_missing_flightnos(flights: List[Flight]) -> List[Flight]:
        for i, f in enumerate(flights):
            if not f.flight_no and i > 0:
                f.flight_no = flights[i - 1].flight_no
        return flights

    def _calculate_quality_score(
        self,
        flights: List[Flight],
        ocr_data: Dict,
        layout_info: Dict,
        quality_metrics: Dict,
        all_warnings: List[str],
    ) -> float:
        """Calculate overall extraction quality score"""
        scores = []

        # Image quality component
        scores.append(quality_metrics["overall"])

        # Extraction completeness
        if ocr_data["flight_numbers"]:
            ocr_flight_count = len(ocr_data["flight_numbers"])
            extraction_ratio = min(1.0, len(flights) / max(1, ocr_flight_count))
            scores.append(extraction_ratio)

        # Field completeness
        field_scores = []
        for flight in flights:
            fields = ["origin", "dest", "sched_out_local", "sched_in_local"]
            filled = sum(1 for f in fields if getattr(flight, f))
            field_scores.append(filled / len(fields))

        if field_scores:
            scores.append(np.mean(field_scores))

        # Confidence average
        if flights:
            avg_conf = sum(f.confidence for f in flights) / len(flights)
            scores.append(avg_conf)

        # Validation score
        warning_factor = len(all_warnings) / max(1, len(flights))
        scores.append(1.0 - 0.1 * warning_factor)


        return np.mean(scores) if scores else 0.0


@app.post("/extract", response_model=Result)
async def extract_flights(file: UploadFile = File(...)):
    """Extract flights from uploaded schedule image with validation"""

    # Validate file
    if not file.content_type.startswith("image"):
        raise HTTPException(415, "Please upload an image file")

    # Read and decode image
    image_data = await file.read()
    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(422, "Unable to process image")

    logger.info(f"Processing image: {file.filename}, size: {img.shape}")

    # Use ensemble extractor
    extractor = EnsembleExtractor()
    result = await extractor.extract_flights(img)

    logger.info(
        f"Extraction complete: {result.total_flights_found} flights, "
        f"quality: {result.quality_score:.2f}, time: {result.processing_time_ms}ms"
    )
    
    # ========== NEW VALIDATION STEP ==========
    if result.flights:
        try:
            # Convert to dict for validation
            result_dict = result.dict()
            
            # Validate and enrich flights
            validated_result = await validate_extraction_results(result_dict)
            
            # Add validation info to result
            if "validation" in validated_result:
                result.validation_warnings.extend(
                    validated_result["validation"].get("warnings", [])[:5]
                )
                
                # Update quality score
                result.quality_score = validated_result.get("quality_score", result.quality_score)
                
                # Add validation metadata
                if not result.schedule_metadata:
                    result.schedule_metadata = {}
                result.schedule_metadata["validation"] = {
                    "valid_flights": validated_result["validation"]["valid_flights"],
                    "average_confidence": validated_result["validation"]["average_confidence"],
                    "sources_used": validated_result["validation"]["sources_used"]
                }
                
                # Enrich flights with validated data
                if "enriched_flights" in validated_result:
                    enriched_map = {
                        f["flight_no"]: f for f in validated_result["enriched_flights"]
                    }
                    
                    for flight in result.flights:
                        enriched = enriched_map.get(flight.flight_no)
                        if not enriched:
                            continue
                        if enriched:
                            # Update with corrections
                            if enriched.get("validation_result", {}).get("corrections"):
                                for field, value in enriched["validation_result"]["corrections"].items():
                                    if hasattr(flight, field):
                                        setattr(flight, field, value)
                            
                            # Add enriched data
                            if enriched.get("validation_result", {}).get("enriched_data"):
                                for field, value in enriched["validation_result"]["enriched_data"].items():
                                    if field in ["origin", "dest"] and not getattr(flight, field):
                                        setattr(flight, field, value)
                                    elif field == "aircraft_type" and not flight.equipment:
                                        flight.equipment = value
                
            logger.info(f"Validation complete: {validated_result.get('validation', {}).get('valid_flights', 0)} valid flights")
            
        except Exception as e:
            logger.error(f"Validation error: {e}")
            # Continue without validation if it fails
            result.validation_warnings.append(f"Validation service unavailable: {str(e)}")
    
    return result

@app.post("/validate")
async def validate_flights(flights: List[Flight]):
    """Validate a list of flights against external APIs"""
    flight_dicts = [f.dict() for f in flights]
    return await validate_flights_endpoint(flight_dicts)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    validation_apis = []
    
    if os.getenv("FLIGHTAWARE_API_KEY"):
        validation_apis.append("flightaware_aeroapi")
    if os.getenv("FLIGHTRADAR24_API_KEY"):
        validation_apis.append("flightradar24")
    
    return {
        "status": "healthy",
        "version": "4.0",
        "model": MODEL,
        "ocr_engines": ["tesseract", "easyocr"] if OCR_AVAILABLE else [],
        "validation_apis": validation_apis,
        "features": [
            "multi-engine-ocr",
            "layout-detection",
            "quality-assessment",
            "relationship-extraction",
            "ensemble-methods",
            "aviation-validation",
            "flight-validation"  # NEW
        ],
    }


# ---------- Run the app ---------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
