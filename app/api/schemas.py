from pydantic import BaseModel

class StatusResponse(BaseModel):
    people: int
    faces: int
    occupied: bool
    appliance_on: bool
    waste_detected: bool
    brightness: int
    latency: float

class MetricsResponse(BaseModel):
    precision: float
    recall: float
    f1_score: float
    false_trigger_rate: float