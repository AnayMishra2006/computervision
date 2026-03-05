"""
Shared in-process state dictionaries.

Both the CV processing loop (app/main.py) and the FastAPI handlers
(app/api/server.py) run in the same Python process but in different
threads.  Using plain dicts with `.update()` is sufficient here because
CPython's GIL makes dict updates effectively atomic for our use case.

latest_state  – per-frame detection results (updated ~every frame)
latest_metrics – running precision / recall / F1 scores
"""

latest_state: dict = {
    "people": 0,
    "faces": 0,
    "occupied": False,
    "appliance_on": False,
    "waste_detected": False,
    "brightness": 0,
    "latency": 0.0,
    "energy_saved_kwh": 0.0,   # cumulative energy saved since startup
}

latest_metrics: dict = {
    "precision": 0.0,
    "recall": 0.0,
    "f1_score": 0.0,
    "false_trigger_rate": 0.0,
}