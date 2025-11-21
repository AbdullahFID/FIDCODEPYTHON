from __future__ import annotations

import os
import logging

import uvicorn
from api import app  # this import will run configure_logging() inside api.py

# Get a module-specific logger instead of importing one from logging_utils
logger = logging.getLogger("flightintel.main")


if __name__ == "__main__":
    host = os.environ.get("FLIGHTINTEL_HOST", "0.0.0.0")
    port = int(os.environ.get("FLIGHTINTEL_PORT", "8001"))

    logger.info("Launching Flight-Intel server on %s:%s", host, port)

    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=os.environ.get("FLIGHTINTEL_RELOAD", "0") == "1",
    )
