# pdf_processor.py
from typing import List

import cv2
import numpy as np
from fastapi import HTTPException
from pdf2image import convert_from_bytes

from config import MAX_WORKERS, thread_pool
from logging_utils import get_logger
logger = get_logger("pdf_processor")
import functools as _functools
import asyncio


class PDFProcessor:
    @staticmethod
    async def convert(pdf_bytes: bytes) -> List[np.ndarray]:
        logger.start_timer("pdf_conversion")
        try:
            pil_images = await asyncio.get_event_loop().run_in_executor(
                thread_pool,
                _functools.partial(
                    convert_from_bytes,
                    pdf_bytes,
                    dpi=300,
                    fmt="PNG",
                    thread_count=min(4, MAX_WORKERS),
                    use_pdftocairo=True,
                ),
            )
            cv_images: List[np.ndarray] = []
            for pil_img in pil_images:
                arr = np.array(pil_img)
                if len(arr.shape) == 2:
                    cv_img = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
                else:
                    cv_img = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                    kernel = np.array(
                        [[0, -0.5, 0], [-0.5, 3, -0.5], [0, -0.5, 0]], dtype=np.float32
                    )
                    cv_img = cv2.filter2D(cv_img, -1, kernel)
                cv_images.append(cv_img)
            logger.end_timer("pdf_conversion")
            logger.logger.info(f"Converted {len(cv_images)} PDF pages")
            return cv_images
        except Exception as e:
            logger.end_timer("pdf_conversion")
            raise HTTPException(422, f"PDF processing failed: {e}")
