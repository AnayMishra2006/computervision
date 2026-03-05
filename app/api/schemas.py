from pydantic import BaseModel


class StatusResponse(BaseModel):
    """Live snapshot of what the CV module currently sees in the room."""
    people: int
    faces: int
    occupied: bool
    appliance_on: bool
    waste_detected: bool
    brightness: int
    latency: float


class MetricsResponse(BaseModel):
    """Real-time performance metrics of the detection pipeline."""
    precision: float
    recall: float
    f1_score: float
    false_trigger_rate: float


class EnergySavingsResponse(BaseModel):
    """Cumulative energy savings calculated since the module started."""
    total_waste_seconds: int        # seconds the room was wasting energy
    energy_saved_kwh: float         # kWh that would have been wasted
    cost_saved: float               # ₹ / $ saved at the configured rate