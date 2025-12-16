# config.py
import os
from concurrent.futures import ThreadPoolExecutor
from openai import AsyncOpenAI

import logging
from logging_utils import configure_logging

configure_logging()
logger = logging.getLogger("flightintel.config")

OPENAI_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_KEY:
    raise RuntimeError("OPENAI_API_KEY missing")

MODEL = os.getenv("FLIGHT_INTEL_MODEL", "gpt-5.1")
MAX_TOKENS = 4096
OPENAI_TIMEOUT = 30

MAX_WORKERS = min(32, (os.cpu_count() or 4) * 2)
thread_pool = ThreadPoolExecutor(max_workers=MAX_WORKERS)

client = AsyncOpenAI(api_key=OPENAI_KEY)

logger.info(
    f"Config: model={MODEL}, timeout={OPENAI_TIMEOUT}s, workers={MAX_WORKERS}"
)
