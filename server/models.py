# models.py
from datetime import datetime
import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class Flight(BaseModel):
    date: str = Field(..., description="MM/DD/YYYY")
    flight_no: str = Field(..., description="Flight number")
    origin: Optional[str] = None
    dest: Optional[str] = None
    sched_out_local: Optional[str] = None
    sched_in_local: Optional[str] = None
    page_number: Optional[int] = None
    confidence: float = Field(default=1.0, ge=0, le=1)

    @field_validator("flight_no")
    @classmethod
    def _clean_flight_no(cls, v: str) -> str:
        if not v:
            return v
        return re.sub(r"[^\w\d]", "", v.upper())

    @field_validator("origin", "dest")
    @classmethod
    def _validate_airport(cls, v: Optional[str]) -> Optional[str]:
        return v.upper() if v and len(v) == 3 else v

    @field_validator("date")
    @classmethod
    def _validate_date(cls, v: str) -> str:
        if v and "/" in v:
            parts = v.split("/")
            if len(parts) == 3:
                m, d, y = parts
                if len(y) == 2:
                    y = f"20{y}"
                return f"{m.zfill(2)}/{d.zfill(2)}/{y}"
        return v


class Result(BaseModel):
    flights: List[Flight]
    connections: List[Dict[str, Any]] = []
    total_flights_found: int = 0
    avg_confidence: float = 0.0
    processing_time: Dict[str, float] = {}
    extraction_method: str = ""


class ExtractionError(BaseModel):
    error: bool = True
    user_message: str
    technical_reason: str
    suggestions: List[str] = Field(default_factory=list)
