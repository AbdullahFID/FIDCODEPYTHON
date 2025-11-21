from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ValidationResult(BaseModel):
    is_valid: bool
    confidence: float
    source: str
    warnings: List[str] = Field(default_factory=list)
    corrections: Dict[str, Any] = Field(default_factory=dict)
    filled_fields: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        validate_assignment = False
        use_enum_values = True


class EnrichedFlight(BaseModel):
    date: str
    flight_no: str
    origin: Optional[str] = None
    dest: Optional[str] = None
    sched_out_local: Optional[str] = None
    sched_in_local: Optional[str] = None
    validation_result: Optional[ValidationResult] = None

    class Config:
        validate_assignment = False
        use_enum_values = True
