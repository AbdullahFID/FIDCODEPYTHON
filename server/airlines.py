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
