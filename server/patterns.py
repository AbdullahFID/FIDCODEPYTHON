# patterns.py
import re


class Patterns:
    FLIGHT_NO = re.compile(r"\b(?:[A-Z]{1,2}\d{1,5}[A-Z]?|[A-Z]\d{5}[A-Z]?|\d{3,5})\b")
    DATE_DMY = re.compile(r"\b(\d{1,2})[/-](\d{1,2})(?:[/-](\d{2,4}))?\b")
    DATE_MDY = re.compile(r"\b(\d{1,2})/([A-Za-z]{3})\b")
    TIME_24H = re.compile(r"\b([01]?\d|2[0-3]):?([0-5]\d)\b")
    AIRPORT = re.compile(r"\b[A-Z]{3}\b")
    ROUTE = re.compile(r"\b([A-Z]{3})\s*[-–→>]\s*([A-Z]{3})\b")


patterns = Patterns()
