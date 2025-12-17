# config.py
import os
from dotenv import load_dotenv
load_dotenv()

from concurrent.futures import ThreadPoolExecutor
import google.generativeai as genai

import logging
from logging_utils import configure_logging

configure_logging()
logger = logging.getLogger("flightintel.config")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    logger.warning("GOOGLE_API_KEY missing - Gemini calls will fail")
else:
    genai.configure(api_key=GOOGLE_API_KEY)

MODEL = os.getenv("FLIGHT_INTEL_MODEL", "gemini-3-flash-preview")
MAX_TOKENS = 8192
TIMEOUT = 30

MAX_WORKERS = min(32, (os.cpu_count() or 4) * 2)
thread_pool = ThreadPoolExecutor(max_workers=MAX_WORKERS)

# No global client needed for genai, we use the module or model instance

logger.info(
    f"Config: model={MODEL}, timeout={TIMEOUT}s, workers={MAX_WORKERS}"
)
