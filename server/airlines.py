# airlines.py
# ---------------------------------------------------------------------
# A *small* subset of the world’s airlines – just the ones you listed.
# Feel free to extend; the rest of the code never needs to change.

AIRLINE_CODES: dict[str, dict[str, str]] = {
    # ==== U.S. MAJOR CARRIERS ====
    "AA": {"icao": "AAL", "name": "American Airlines"},
    "UA": {"icao": "UAL", "name": "United Airlines"},
    "DL": {"icao": "DAL", "name": "Delta Air Lines"},
    "WN": {"icao": "SWA", "name": "Southwest Airlines"},
    "B6": {"icao": "JBU", "name": "JetBlue Airways"},
    "AS": {"icao": "ASA", "name": "Alaska Airlines"},
    "NK": {"icao": "NKS", "name": "Spirit Airlines"},
    "F9": {"icao": "FFT", "name": "Frontier Airlines"},
    "G4": {"icao": "AAY", "name": "Allegiant Air"},
    "SY": {"icao": "SCX", "name": "Sun Country Airlines"},
    "HA": {"icao": "HAL", "name": "Hawaiian Airlines"},
    "MX": {"icao": "MXY", "name": "Breeze Airways"},

    # ==== U.S. CARGO ====
    "5X": {"icao": "UPS", "name": "UPS Airlines"},
    "FX": {"icao": "FDX", "name": "FedEx Express"},
    "5Y": {"icao": "GTI", "name": "Atlas Air"},
    "K4": {"icao": "CKS", "name": "Kalitta Air"},
    "NC": {"icao": "NAC", "name": "Northern Air Cargo"},
    "GB": {"icao": "ABX", "name": "ABX Air"},
    "3S": {"icao": "PAC", "name": "Polar Air Cargo"},
    "M6": {"icao": "AJT", "name": "Amerijet International"},
    "CV": {"icao": "CLX", "name": "Cargolux"},
    "KZ": {"icao": "NCA", "name": "Nippon Cargo Airlines"},
    "PO": {"icao": "PAC", "name": "Polar Air Cargo"},

    # ==== U.S. REGIONAL ====
    "OO": {"icao": "SKW", "name": "SkyWest Airlines"},
    "9E": {"icao": "EDV", "name": "Endeavor Air"},
    "ZW": {"icao": "AWI", "name": "Air Wisconsin"},
    "YV": {"icao": "ASH", "name": "Mesa Airlines"},
    "YX": {"icao": "RPA", "name": "Republic Airways"},
    "OH": {"icao": "JIA", "name": "PSA Airlines"},
    "MQ": {"icao": "ENY", "name": "Envoy Air"},
    "G7": {"icao": "GJS", "name": "GoJet Airlines"},
    "CP": {"icao": "CPZ", "name": "Compass Airlines"},
    "PT": {"icao": "PDT", "name": "Piedmont Airlines"},
    "9K": {"icao": "KAP", "name": "Cape Air"},
    "C5": {"icao": "UCA", "name": "CommutAir"},
    "QX": {"icao": "QXE", "name": "Horizon Air"},
    "AX": {"icao": "LOF", "name": "Trans States Airlines"},
    "EV": {"icao": "ASQ", "name": "ExpressJet Airlines"},
    "S5": {"icao": "TCF", "name": "Shuttle America"},

    # ==== CHARTER / LEISURE ====
    "N0": {"icao": "NBT", "name": "Norse Atlantic Airways"},
    "MN": {"icao": "CAW", "name": "Comair (South Africa)"},
    "XT": {"icao": "CXP", "name": "Xtra Airways"},
    "SX": {"icao": "SKX", "name": "Skyways Express"},
    "EG": {"icao": "JAA", "name": "Japan Asia Airways"},

    # ==== LOW‑COST (legacy) ====
    "VX": {"icao": "VRD", "name": "Virgin America"},
    "FL": {"icao": "TRS", "name": "AirTran Airways"},
    "TZ": {"icao": "AMT", "name": "ATA Airlines"},
    "J7": {"icao": "VJA", "name": "ValuJet"},
    "WP": {"icao": "MKU", "name": "Island Air"},

    # ==== BUSINESS / PRIVATE ====
    "XO": {"icao": "XOJ", "name": "XOJET"},
    "FJ": {"icao": "LXJ", "name": "Flexjet"},
    "NJ": {"icao": "EJA", "name": "NetJets"},
    "WU": {"icao": "UPJ", "name": "Wheels Up"},
    "JJ": {"icao": "JIT", "name": "Jet It"},
    "SJ": {"icao": "URF", "name": "Surf Air"},
}

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

AIRPORT_TIMEZONES = {
    # Eastern Time
    "JFK": "America/New_York", "LGA": "America/New_York", "EWR": "America/New_York",
    "ATL": "America/New_York", "BOS": "America/New_York", "DCA": "America/New_York",
    "MIA": "America/New_York", "MCO": "America/New_York", "PHL": "America/New_York",
    "CLT": "America/New_York", "BWI": "America/New_York", "FLL": "America/New_York",
    "TPA": "America/New_York", "RDU": "America/New_York", "PIT": "America/New_York",
    
    # Central Time
    "ORD": "America/Chicago", "MDW": "America/Chicago", "DFW": "America/Chicago",
    "IAH": "America/Chicago", "MSP": "America/Chicago", "STL": "America/Chicago",
    "MCI": "America/Chicago", "MKE": "America/Chicago", "MSY": "America/Chicago",
    "BNA": "America/Chicago", "AUS": "America/Chicago", "SAT": "America/Chicago",
    "MEM": "America/Chicago", "OKC": "America/Chicago",
    
    # Mountain Time  
    "DEN": "America/Denver", "SLC": "America/Denver", "ABQ": "America/Denver",
    
    # Pacific Time
    "LAX": "America/Los_Angeles", "SFO": "America/Los_Angeles", "SEA": "America/Los_Angeles",
    "SAN": "America/Los_Angeles", "PDX": "America/Los_Angeles", "LAS": "America/Los_Angeles",
    "SJC": "America/Los_Angeles", "OAK": "America/Los_Angeles", "SMF": "America/Los_Angeles",
    "BUR": "America/Los_Angeles", "ONT": "America/Los_Angeles", "SNA": "America/Los_Angeles",
    
    # Arizona (no DST)
    "PHX": "America/Phoenix", "TUS": "America/Phoenix",
    
    # Alaska
    "ANC": "America/Anchorage", "FAI": "America/Anchorage", "JNU": "America/Anchorage",
    
    # Hawaii
    "HNL": "Pacific/Honolulu", "OGG": "Pacific/Honolulu", "KOA": "Pacific/Honolulu",
    "LIH": "Pacific/Honolulu", "ITO": "Pacific/Honolulu",
    
    # Detroit (Eastern but sometimes listed separately)
    "DTW": "America/Detroit",
    
    # Puerto Rico (Atlantic)
    "SJU": "America/Puerto_Rico",
}