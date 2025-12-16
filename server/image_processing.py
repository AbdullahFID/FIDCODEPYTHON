# image_processing.py
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

import logging
logger = logging.getLogger(__name__)



class UltimateImageProcessor:
    """Create multiple optimized views (original / enhanced / sharpened / edges)."""

    @staticmethod
    def analyze_image(img: np.ndarray) -> Dict[str, Any]:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        contrast = gray.std()
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        text_regions = sum(1 for c in contours if cv2.contourArea(c) > 50)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)
        has_grid = lines is not None and len(lines) > 15
        h, w = img.shape[:2]
        aspect_ratio = w / h
        info = {
            "sharpness": lap_var,
            "contrast": contrast,
            "text_regions": text_regions,
            "has_grid": has_grid,
            "is_landscape": aspect_ratio > 1.3,
            "needs_enhancement": lap_var < 100 or contrast < 35,
            "is_very_blurry": lap_var < 50,
            "is_low_contrast": contrast < 25,
        }
        logger.info(
            f"Image analysis: sharp={info['sharpness']:.1f}, "
            f"contrast={info['contrast']:.1f}, grid={info['has_grid']}, "
            f"text_regions={info['text_regions']}"
        )
        return info

    @staticmethod
    def create_optimal_versions(img: np.ndarray) -> List[Tuple[np.ndarray, str]]:
        analysis = UltimateImageProcessor.analyze_image(img)
        versions: List[Tuple[np.ndarray, str]] = [(img, "original")]

        if analysis["needs_enhancement"]:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            versions.append((cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR), "enhanced"))

            if analysis["is_very_blurry"]:
                denoised = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)
                kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
                sharpened = cv2.filter2D(denoised, -1, kernel)
                versions.append((cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR), "sharpened"))

            if analysis["is_low_contrast"]:
                _, binary = cv2.threshold(
                    enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
                )
                versions.append((cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR), "binary"))

        if analysis["has_grid"]:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            versions.append((cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR), "edges"))

        return versions
