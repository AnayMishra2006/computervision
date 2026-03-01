from fastapi import FastAPI
from app.api.state import latest_state, latest_metrics
from app.api.schemas import StatusResponse, MetricsResponse

app = FastAPI(title="Watt-Watch API", version="1.0")

# ✅ Health check
@app.get("/")
def root():
    return {"message": "Watt-Watch API running"}

# ✅ Get live system status
@app.get("/status", response_model=StatusResponse)
def get_status():
    return latest_state

# ✅ Get metrics
@app.get("/metrics", response_model=MetricsResponse)
def get_metrics():
    return latest_metrics