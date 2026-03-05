# app/main.py
"""
Main computer-vision processing loop for Watt-Watch.

This module ties together every sub-system:
  1. Camera capture (webcam / RTSP / file)
  2. People detection  (YOLOv8 body + Haar face – belt-and-braces approach)
  3. Appliance detection (ceiling ROI brightness analysis)
  4. Privacy layer (Gaussian blur over every detected person)
  5. Waste logic engine (rule: empty room + lights ON → alert)
  6. Metrics (online Precision / Recall / F1)
  7. MQTT publish to ESP32 (only when state changes, with error tolerance)
  8. Shared frame store (so the FastAPI /stream and /frame endpoints work)

The loop runs in its own daemon thread (started by run.py) so the
FastAPI server can serve API requests concurrently.
"""

import time
import cv2

from app.cv.camera import get_camera
from app.cv.detector import detect_people
from app.cv.appliance import detect_appliance
from app.cv.privacy import blur_people
from app.cv.face_detector import detect_faces
from app.cv import frame_store

from app.logic.engine import WasteDetector
from app.mqtt.client import MQTTClient
from app.metrics.evaluator import Evaluator

from app.config import (
    FRAME_WIDTH, FRAME_HEIGHT, JPEG_QUALITY,
    MQTT_TOPIC_LIGHTS, MQTT_TOPIC_FAN,
    LIGHT_WATTAGE_W, FAN_WATTAGE_W,
)
from app.api.state import latest_state, latest_metrics

# ---------------------------------------------------------------------------
# Module-level singletons (created once, reused every frame)
# ---------------------------------------------------------------------------
logic = WasteDetector()       # rule engine with debounce
mqtt = MQTTClient()           # ESP32 MQTT publisher
evaluator = Evaluator()       # online metrics tracker

# Running total of energy saved (kWh) since the process started
_energy_saved_kwh: float = 0.0
_waste_start_time: float | None = None  # when the current waste period began


def _draw_overlay(frame, person_count, face_count, brightness,
                  appliance_on, waste_detected, latency, metrics):
    """
    Render status text onto *frame* for the local preview window.

    This overlay is written AFTER blurring so the text is always legible.
    Text positions are spread out vertically to avoid overlap.
    """
    status_text  = "⚡ WASTE DETECTED" if waste_detected else "✅ SECURE"
    status_color = (0, 0, 255)       if waste_detected else (0, 255, 0)

    overlay_items = [
        (f"People (YOLO): {person_count}",          (0, 255, 0),   30),
        (f"Faces (Haar):  {face_count}",             (255, 255, 0), 60),
        (f"Brightness:    {int(brightness)}",        (255, 255, 255), 90),
        (f"Appliance ON:  {appliance_on}",           (255, 255, 255), 120),
        (status_text,                                 status_color,  155),
        (f"Latency: {latency:.2f}s",                 (0, 255, 255), 185),
        (f"F1: {metrics['F1 Score']:.2f}  "
         f"P: {metrics['Precision']:.2f}  "
         f"R: {metrics['Recall']:.2f}",              (255, 255, 0), 215),
    ]

    for text, color, y in overlay_items:
        cv2.putText(frame, text, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)


def main():
    """
    Infinite capture-and-process loop.  Exits on ESC key or camera failure.
    """
    global _energy_saved_kwh, _waste_start_time

    cap = get_camera()
    prev_light_state = None
    prev_fan_state   = None

    while True:
        ret, frame = cap.read()

        # --- Check for camera failure immediately after read ---
        if not ret:
            print("⚠️  Camera read failed – retrying in 1 second…")
            time.sleep(1)
            continue

        loop_start = time.time()

        # ------------------------------------------------------------------
        # 1. Pre-process: resize + JPEG compress/decompress
        #    JPEG round-trip intentionally mimics the quality degradation
        #    of a real CCTV stream compressed at the camera head.
        # ------------------------------------------------------------------
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
        _, enc = cv2.imencode(".jpg", frame, encode_param)
        frame = cv2.imdecode(enc, cv2.IMREAD_COLOR)

        # Mirror for webcam (disable for real CCTV where mirroring is wrong)
        frame = cv2.flip(frame, 1)

        # ------------------------------------------------------------------
        # 2. Detection
        # ------------------------------------------------------------------
        person_count, body_boxes = detect_people(frame)
        appliance_on, brightness = detect_appliance(frame)
        face_count, _face_rects  = detect_faces(frame)

        # Combine body and face signals: room is occupied if either fires
        occupied             = (person_count > 0) or (face_count > 0)
        effective_count      = 1 if occupied else 0

        # ------------------------------------------------------------------
        # 3. Privacy layer – blur people BEFORE the frame is stored/streamed
        # ------------------------------------------------------------------
        frame = blur_people(frame, body_boxes)

        # ------------------------------------------------------------------
        # 4. Rule engine
        # ------------------------------------------------------------------
        waste_detected = logic.update(effective_count, appliance_on)

        # ------------------------------------------------------------------
        # 5. Energy savings accumulation
        #    Track how long the waste state is continuously active and
        #    accumulate kWh saved (assuming we'd cut the load via IoT).
        # ------------------------------------------------------------------
        now = time.time()
        if waste_detected:
            if _waste_start_time is None:
                _waste_start_time = now
            else:
                # Seconds the waste has been active in this burst
                wasted_seconds = now - _waste_start_time
                # kWh = (W × h) / 1000
                total_w = LIGHT_WATTAGE_W + FAN_WATTAGE_W
                _energy_saved_kwh = (total_w * (wasted_seconds / 3600)) / 1000
        else:
            _waste_start_time = None

        # ------------------------------------------------------------------
        # 6. Metrics
        # ------------------------------------------------------------------
        # Auto ground-truth: waste condition = empty room + appliance ON
        ground_truth = 1 if (not occupied and appliance_on) else 0
        evaluator.update(ground_truth, int(waste_detected))
        metrics = evaluator.compute()

        latency = time.time() - loop_start

        # ------------------------------------------------------------------
        # 7. Update shared state for API endpoints
        # ------------------------------------------------------------------
        latest_state.update({
            "people":           person_count,
            "faces":            face_count,
            "occupied":         occupied,
            "appliance_on":     appliance_on,
            "waste_detected":   waste_detected,
            "brightness":       int(brightness),
            "latency":          round(latency, 4),
            "energy_saved_kwh": round(_energy_saved_kwh, 6),
        })

        latest_metrics.update({
            "precision":          metrics["Precision"],
            "recall":             metrics["Recall"],
            "f1_score":           metrics["F1 Score"],
            "false_trigger_rate": metrics["False Trigger Rate"],
        })

        # ------------------------------------------------------------------
        # 8. MQTT – publish to ESP32 only when device state changes
        # ------------------------------------------------------------------
        light_state = "OFF" if waste_detected else "ON"
        fan_state   = "OFF" if waste_detected else "ON"

        if light_state != prev_light_state:
            mqtt.publish_command(MQTT_TOPIC_LIGHTS, light_state)
            prev_light_state = light_state

        if fan_state != prev_fan_state:
            mqtt.publish_command(MQTT_TOPIC_FAN, fan_state)
            prev_fan_state = fan_state

        # Also publish waste status for dashboard subscriptions
        mqtt.publish_waste("room101", waste_detected)

        # ------------------------------------------------------------------
        # 9. Overlay text + store anonymised frame for API /stream endpoint
        # ------------------------------------------------------------------
        _draw_overlay(frame, person_count, face_count, brightness,
                      appliance_on, waste_detected, latency, metrics)

        # Encode the annotated, blurred frame to JPEG and push to frame store
        _, jpeg_buf = cv2.imencode(".jpg", frame, encode_param)
        frame_store.set_frame(jpeg_buf.tobytes())

        # ------------------------------------------------------------------
        # 10. Optional local preview window (works when a display is present)
        # ------------------------------------------------------------------
        try:
            cv2.imshow("Watt-Watch | Live Preview", frame)
            if cv2.waitKey(1) == 27:  # ESC to quit
                break
        except cv2.error:
            # No display available (headless server / CI) – just keep looping
            pass

        # Console status line (non-blocking, overwrites same line)
        print(
            f"\r👁  People: {person_count}  Faces: {face_count}  "
            f"Lights: {'ON' if appliance_on else 'OFF'}  "
            f"Waste: {waste_detected}  "
            f"F1: {metrics['F1 Score']:.2f}  "
            f"Latency: {latency*1000:.0f}ms",
            end="",
            flush=True,
        )

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()