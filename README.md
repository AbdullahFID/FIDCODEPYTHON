# FIDCODEPYTHON

Simple demo for the Flight‑Intel extractor. To use:

1. Install dependencies:
   ```bash
   pip install -r server/requirements.txt
   ```

2. Start the API server:
   ```bash
   cd server
   python main.py
   ```

3. Open `client/index.html` in your browser and upload a roster image or PDF.

┌─────────────────────────────────────────────────────────────────────┐
│                       API REQUEST RECEIVED                           │
│                    (Image with Flight Data)                          │
└─────────────────────────────────────┬───────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      IMAGE PROCESSING                                │
│                  (OCR/GPT Extraction)                               │
│    • Extract flight numbers, dates, times, airports                 │
│    • Parse schedule format                                          │
│    • Initial quality score                                          │
└─────────────────────────────────────┬───────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    FLIGHT DATA VALIDATION                           │
│                  validate_extraction_results()                      │
└─────────────────────────────────────┬───────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────┐
│                CHECK EXISTING FIELDS & CACHE                        │
│    • All fields present? → Skip API calls                          │
│    • Check validation cache (15 min TTL)                           │
│    • Identify missing fields                                       │
└─────────────────┬───────────────────────────────┬───────────────────┘
                  │                               │
                  ▼                               ▼
         [Fields Complete]                 [Missing Fields]
                  │                               │
                  ▼                               ▼
┌─────────────────────────┐    ┌─────────────────────────────────────┐
│   RETURN VALIDATED      │    │         CHOOSE VALIDATION PATH      │
│   confidence: 1.0       │    │  • Live (≤2 days): /flights/{id}   │
│   source: "prefilled"   │    │  • Schedule (>2 days): /schedules   │
└─────────────────────────┘    └──────────────┬──────────────────────┘
                                              │
                                              ▼
                              ┌───────────────────────────────┐
                              │    PARALLEL API CALLS         │
                              └───────────┬───────────────────┘
                                         │
        ┌────────────────────────────────┼────────────────────────────┐
        ▼                                ▼                             ▼
┌─────────────────┐           ┌─────────────────┐          ┌─────────────────┐
│  FLIGHTAWARE    │           │  FLIGHTRADAR24  │          │    FIREHOSE     │
│   AEROAPI V4    │           │      API        │          │   (REALTIME)    │
├─────────────────┤           ├─────────────────┤          ├─────────────────┤
│ • Origin/Dest   │           │ • Flight status │          │ • Live updates  │
│ • Schedule times│           │ • Aircraft type │          │ • Gate info     │
│ • Flight plan   │           │ • Route info    │          │ • Actual times  │
│ • Confidence:   │           │ • Confidence:   │          │ • Confidence:   │
│   0.95          │           │   0.85          │          │   0.99          │
└────────┬────────┘           └────────┬────────┘          └────────┬────────┘
         │                             │                             │
         └─────────────────────────────┴─────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA PROCESSING                              │
│  • Extract & normalize airport codes (IATA/ICAO)                   │
│  • Convert times to local timezone (HHMM format)                   │
│  • Fill missing fields (origin, dest, times)                       │
│  • Apply corrections for existing fields                           │
│  • Cross-validate between sources                                  │
└─────────────────────────────────────┬───────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     ENRICHED FLIGHT OBJECT                         │
│  {                                                                 │
│    "flight_no": "AA123",                                          │
│    "date": "08/15/2025",                                          │
│    "origin": "JFK",        // filled/corrected                    │
│    "dest": "LAX",          // filled/corrected                    │
│    "sched_out_local": "0830",                                     │
│    "sched_in_local": "1145",                                      │
│    "validation_result": {                                          │
│      "is_valid": true,                                            │
│      "confidence": 0.95,                                          │
│      "source": "aeroapi+fr24+firehose",                          │
│      "filled_fields": {...},                                      │
│      "corrections": {...}                                         │
│    }                                                              │
│  }                                                                │
└─────────────────────────────────────┬───────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      VALIDATION SUMMARY                             │
│  • Total flights processed                                          │
│  • Valid flights count                                             │
│  • Average confidence score                                        │
│  • Fields filled/corrected                                         │
│  • Processing time                                                 │
│  • Sources used                                                    │
│  • Quality score (70% OCR + 30% validation)                       │
└─────────────────────────────────────┬───────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    SEND REPORT TO BOSS                             │
│  • Enriched flight schedule with all missing data filled          │
│  • Validation confidence metrics                                   │
│  • Data source attribution                                         │
│  • Processing statistics                                           │
│  • Any warnings or conflicts found                                │
└─────────────────────────────────────────────────────────────────────┘
