"""
Pydantic response schemas for the Watt-Watch API.
"""
from pydantic import BaseModel


class StatusResponse(BaseModel):
    """Live room status snapshot."""
    room_id:          str
    people:           int
    faces:            int
    occupied:         bool
    appliance_on:     bool
    waste_detected:   bool
    brightness:       int
    latency:          float
    energy_saved_wh:  float   # cumulative watt-hours saved this session


class MetricsResponse(BaseModel):
    """Detection-quality metrics (self-evaluated, real-time)."""
    precision:          float
    recall:             float
    f1_score:           float
    false_trigger_rate: float


class HealthResponse(BaseModel):
    """Simple health-check payload."""
    status:  str
    version: str
