# app/api/schemas.py
"""
Pydantic response models for the FastAPI endpoints.

These models define the exact JSON shape that your website friend's
front-end will receive when it calls the API.  Any field added here
must also be populated in app/api/state.py and app/main.py.
"""

from pydantic import BaseModel, Field


class StatusResponse(BaseModel):
    """Live room status – returned by GET /status."""
    people: int = Field(description="Number of people detected by YOLO")
    faces: int = Field(description="Number of faces detected by Haar cascade")
    occupied: bool = Field(description="True if room is considered occupied")
    appliance_on: bool = Field(description="True if lights/screens detected as ON")
    waste_detected: bool = Field(description="True if energy waste alert is active")
    brightness: int = Field(description="Mean pixel brightness of ceiling ROI (0-255)")
    latency: float = Field(description="End-to-end processing latency in seconds")
    energy_saved_kwh: float = Field(description="Cumulative energy saved since startup (kWh)")


class MetricsResponse(BaseModel):
    """Running detection performance metrics – returned by GET /metrics."""
    precision: float = Field(description="Precision of waste detection")
    recall: float = Field(description="Recall of waste detection")
    f1_score: float = Field(description="F1 score (harmonic mean of precision & recall)")
    false_trigger_rate: float = Field(description="Rate of false waste alerts")