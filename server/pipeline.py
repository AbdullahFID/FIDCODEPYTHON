# pipeline.py
from typing import Dict, List

import cv2
import numpy as np

from connections import ConnectionDetector
from extraction_engine import PerfectExtractionEngine
from image_processing import UltimateImageProcessor
from logging_utils import get_logger
logger = get_logger("pipeline")
from models import Flight, Result


class UltimatePipeline:
    def __init__(self) -> None:
        self.processor = UltimateImageProcessor()
        self.extractor = PerfectExtractionEngine()
        self.connector = ConnectionDetector()

    async def process(self, images: List[np.ndarray]) -> Result:
        logger.start_timer("complete_pipeline")
        timing: Dict[str, float] = {}
        all_flights: List[Flight] = []

        for page_num, img in enumerate(images, 1):
            page_key = f"page_{page_num}"
            prep_key = f"{page_key}_prep"
            extract_key = f"{page_key}_extract"

            logger.start_timer(page_key)
            logger.start_timer(prep_key)

            versions = self.processor.create_optimal_versions(img)
            encoded_versions: List[tuple[str, str]] = []
            for img_array, vtype in versions:
                ok, buffer = cv2.imencode(".png", img_array, [cv2.IMWRITE_PNG_COMPRESSION, 5])
                if ok:
                    b64 = np.frombuffer(buffer, dtype=np.uint8)
                    # convert to base64 string
                    import base64

                    encoded_versions.append(
                        (base64.b64encode(buffer).decode("utf-8"), vtype)
                    )

            timing[prep_key] = logger.end_timer(prep_key)

            logger.start_timer(extract_key)
            page_flights = await self.extractor.extract_comprehensive(encoded_versions)
            timing[extract_key] = logger.end_timer(extract_key)

            for f in page_flights:
                f.page_number = page_num
            all_flights.extend(page_flights)

            timing[f"{page_key}_total"] = logger.end_timer(page_key)
            logger.logger.info(f"Page {page_num}: found {len(page_flights)} flights")

        logger.start_timer("connections")
        connections = self.connector.find_connections(all_flights)
        timing["connections"] = logger.end_timer("connections")

        avg_conf = (
            sum(f.confidence for f in all_flights) / len(all_flights)
            if all_flights
            else 0.0
        )
        timing["total"] = logger.end_timer("complete_pipeline")

        if all_flights:
            methods = set()
            if any(f.confidence >= 0.9 for f in all_flights):
                methods.add("direct")
            if any(0.7 <= f.confidence < 0.9 for f in all_flights):
                methods.add("enhanced")
            if any(f.confidence < 0.7 for f in all_flights):
                methods.add("forensic")
            method = "+".join(sorted(methods))
        else:
            method = "none"

        return Result(
            flights=all_flights,
            connections=connections,
            total_flights_found=len(all_flights),
            avg_confidence=avg_conf,
            processing_time=timing,
            extraction_method=method,
        )
