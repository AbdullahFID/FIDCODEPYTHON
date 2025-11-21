# main.py
import uvicorn

from api import app
from logging_utils import logger


if __name__ == "__main__":
    logger.logger.info("Starting Flight-Intel v8.1 server")
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8001,
        log_level="info",
        access_log=True,
    )
