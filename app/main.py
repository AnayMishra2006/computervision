"""
Watt-Watch — main CV processing pipeline.

This module is the heart of the system.  It runs in a background thread
(started by run.py) and continuously:

  1. Grabs a frame from the camera (webcam or RTSP CCTV stream).
  2. Detects people using YOLOv8 (accurate even at CCTV distances).
  3. Detects faces with Haar-cascade (catches seated / partially-visible people).
  4. Detects whether appliances (lights, screens) are ON via brightness analysis.
  5. Anonymises the frame (blurs every detected person + face region).
  6. Runs the rule-based WasteDetector logic engine.
  7. Publishes MQTT commands to the ESP32 when device state changes.
  8. Pushes the anonymised JPEG frame + state snapshot to the shared API
     state store so the FastAPI server can serve them in real time.
"""
import cv2
import time
import logging
from app.cv.camera    import get_camera
from app.cv.detector  import detect_people
from app.cv.appliance import detect_appliance
from app.cv.privacy   import anonymise_frame
from app.cv.face_detector import detect_faces
from app.logic.engine import WasteDetector
from app.mqtt.client  import MQTTClient
from app.metrics.evaluator import Evaluator
from app.api import state as st
from app.config import (
    FRAME_WIDTH, FRAME_HEIGHT, JPEG_QUALITY, ROOM_ID, WASTE_DELAY_SECONDS
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level singletons — created once, reused every frame
# ---------------------------------------------------------------------------
logic     = WasteDetector(delay_seconds=WASTE_DELAY_SECONDS)
mqtt      = MQTTClient()
evaluator = Evaluator()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _draw_hud(frame, person_count, face_count, brightness, appliance_on,
              waste_detected, latency, metrics):
    """
    Draw a non-intrusive HUD (Heads-Up Display) on the anonymised frame.
    All text is overlaid *after* blurring so personal identifiers are gone.
    """
    status_text  = "WASTE DETECTED" if waste_detected else "SECURE"
    status_color = (0, 0, 255)      if waste_detected else (0, 255, 0)

    labels = [
        (f"People:    {person_count}",        (0, 255, 0)),
        (f"Faces:     {face_count}",           (255, 255, 0)),
        (f"Appliance: {'ON' if appliance_on else 'OFF'}", (255, 255, 255)),
        (f"Brightness:{int(brightness)}",      (255, 255, 255)),
        (f"Latency:   {latency:.2f}s",         (0, 255, 255)),
        (f"F1:        {metrics['F1 Score']:.2f}", (255, 255, 0)),
        (status_text,                           status_color),
    ]

    for i, (text, color) in enumerate(labels):
        cv2.putText(frame, text, (10, 30 + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)


def _encode_jpeg(frame) -> bytes:
    """Encode a BGR frame to JPEG bytes for the MJPEG stream."""
    params = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
    _, buf  = cv2.imencode(".jpg", frame, params)
    return buf.tobytes()


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    """
    Entry point for the CV thread.

    Runs indefinitely until the user presses ESC (when a display window
    is available) or the process is killed.
    """
    camera = get_camera()

    # Remember previous device states so we only publish MQTT on change
    prev_light_state: str | None = None
    prev_fan_state:   str | None = None

    logger.info("Watt-Watch CV pipeline started (room=%s)", ROOM_ID)

    while True:
        ret, frame = camera.read()
        start_time = time.time()

        if not ret:
            logger.error("Could not read frame — stopping pipeline")
            break

        # ------------------------------------------------------------------
        # 1. Pre-process: resize to target resolution
        # ------------------------------------------------------------------
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        # Mirror view (natural for webcam demos; remove for fixed CCTV mount)
        frame = cv2.flip(frame, 1)

        # ------------------------------------------------------------------
        # 2. Detection
        # ------------------------------------------------------------------
        person_count, person_boxes = detect_people(frame)
        appliance_on, brightness   = detect_appliance(frame)
        face_count,   _            = detect_faces(frame)

        # Occupancy = at least one person detected by YOLO OR face detector
        occupied               = (person_count > 0) or (face_count > 0)
        effective_person_count = 1 if occupied else 0

        # ------------------------------------------------------------------
        # 3. Privacy — anonymise frame BEFORE storing / streaming
        # ------------------------------------------------------------------
        frame = anonymise_frame(frame, person_boxes)

        # ------------------------------------------------------------------
        # 4. Logic engine
        # ------------------------------------------------------------------
        waste_detected = logic.update(effective_person_count, appliance_on)
        latency        = time.time() - start_time

        # ------------------------------------------------------------------
        # 5. Self-evaluation metrics
        #    Ground truth: waste should have been flagged when empty+ON
        # ------------------------------------------------------------------
        ground_truth = 1 if (not occupied and appliance_on) else 0
        evaluator.update(ground_truth, int(waste_detected))
        metrics = evaluator.compute()

        # ------------------------------------------------------------------
        # 6. Update shared API state store
        # ------------------------------------------------------------------
        st.update_state({
            "room_id":          ROOM_ID,
            "people":           person_count,
            "faces":            face_count,
            "occupied":         occupied,
            "appliance_on":     appliance_on,
            "waste_detected":   waste_detected,
            "brightness":       int(brightness),
            "latency":          round(latency, 4),
            "energy_saved_wh":  logic.energy_saved_wh,
        })

        st.update_metrics({
            "precision":          metrics["Precision"],
            "recall":             metrics["Recall"],
            "f1_score":           metrics["F1 Score"],
            "false_trigger_rate": metrics["False Trigger Rate"],
        })

        # ------------------------------------------------------------------
        # 7. MQTT — publish to ESP32 only when device state changes
        # ------------------------------------------------------------------
        light_state = "OFF" if waste_detected else "ON"
        fan_state   = "OFF" if waste_detected else "ON"

        if light_state != prev_light_state:
            mqtt.publish_command(f"wattwatch/{ROOM_ID}/lights/cmd", light_state)
            prev_light_state = light_state

        if fan_state != prev_fan_state:
            mqtt.publish_command(f"wattwatch/{ROOM_ID}/fan/cmd", fan_state)
            prev_fan_state = fan_state

        # Also publish the waste flag so the dashboard can subscribe directly
        mqtt.publish_waste(ROOM_ID, waste_detected)

        # ------------------------------------------------------------------
        # 8. Draw HUD and push to MJPEG frame buffer
        # ------------------------------------------------------------------
        _draw_hud(frame, person_count, face_count, brightness,
                  appliance_on, waste_detected, latency, metrics)

        st.update_frame(_encode_jpeg(frame))

        # ------------------------------------------------------------------
        # 9. Console metrics summary (single-line overwrite)
        # ------------------------------------------------------------------
        print(
            f"\rRoom={ROOM_ID} | People={person_count} | "
            f"Appliance={'ON' if appliance_on else 'OFF'} | "
            f"Waste={'YES' if waste_detected else 'NO'} | "
            f"Latency={latency:.2f}s | "
            f"Energy Saved={logic.energy_saved_wh:.4f}Wh",
            end="",
            flush=True,
        )

        # ------------------------------------------------------------------
        # 10. Optional local preview window (for dev / demo)
        # ------------------------------------------------------------------
        cv2.imshow("Watt-Watch | Press ESC to quit", frame)
        if cv2.waitKey(1) == 27:   # ESC key
            break

    # Cleanup
    camera.release()
    mqtt.stop()
    cv2.destroyAllWindows()
    logger.info("CV pipeline stopped")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
