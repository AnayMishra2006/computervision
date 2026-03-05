# app/main.py
#
# Main computer-vision loop for Watt-Watch.
#
# What happens here every frame:
#   1. Grab frame from camera
#   2. Detect people (YOLO + ByteTrack)
#   3. Detect faces (Haar cascade — secondary occupancy signal)
#   4. Detect appliance / light state (ceiling brightness)
#   5. Apply privacy anonymisation (body blur + face pixelation)
#   6. Run the waste-detection logic engine
#   7. Calculate energy savings
#   8. Publish MQTT commands to the ESP32 (only on state change)
#   9. Update shared state so the FastAPI endpoints have fresh data
#  10. Encode the anonymised frame for the /stream endpoint
#  11. Optionally show a local display window (disabled in HEADLESS mode)

import cv2
import time

from app.cv.camera      import get_camera
from app.cv.detector    import detect_people
from app.cv.appliance   import detect_appliance
from app.cv.privacy     import blur_people, pixelate_faces
from app.cv.face_detector import detect_faces
from app.logic.engine   import WasteDetector
from app.mqtt.client    import MQTTClient
from app.metrics.evaluator import Evaluator
import app.api.state as shared

from app.config import (
    FRAME_WIDTH, FRAME_HEIGHT, JPEG_QUALITY,
    HEADLESS, MQTT_ROOM,
)

# ---------------------------------------------------------------------------
# Module-level singletons (created once, reused every frame)
# ---------------------------------------------------------------------------
_logic    = WasteDetector()
_mqtt     = MQTTClient()
_evaluator = Evaluator()


# ---------------------------------------------------------------------------
# Overlay helpers
# ---------------------------------------------------------------------------

def _draw_boxes(frame, boxes):
    """Draw green bounding boxes around detected people."""
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)


def _draw_overlay(frame, state: dict, metrics: dict):
    """
    Render all HUD text onto the frame.
    Kept in one place so it is easy to modify without touching detection logic.
    """
    person_count  = state["people"]
    face_count    = state["faces"]
    appliance_on  = state["appliance_on"]
    waste_detected = state["waste_detected"]
    brightness    = state["brightness"]
    latency       = state["latency"]
    f1            = metrics["F1 Score"]

    # People + faces
    cv2.putText(frame, f"People: {person_count}",   (10,  30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f"Faces:  {face_count}",     (10,  60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(frame, f"Brightness: {brightness}", (10,  90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f"Appliance: {'ON' if appliance_on else 'OFF'}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f"Latency: {latency:.2f}s",  (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(frame, f"F1: {f1:.2f}",             (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # Waste status banner
    status_text  = "WASTE DETECTED" if waste_detected else "SECURE"
    status_color = (0, 0, 255)      if waste_detected else (0, 255, 0)
    cv2.putText(frame, status_text, (10, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 2)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    """
    Entry point for the CV pipeline.
    Called from run.py in a background thread while the FastAPI server
    runs in the main thread.
    """
    cap = get_camera()

    # Track previous MQTT states so we only publish on changes
    prev_light_state = None
    prev_fan_state   = None

    while True:
        ret, frame = cap.read()
        if not ret:
            # Camera disconnected or video file ended — stop gracefully
            print("⚠️  Camera read failed. Stopping CV loop.")
            break

        start_time = time.time()

        # Resize to configured resolution
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

        # JPEG compression round-trip: simulates the lossy encoding that real
        # CCTV cameras apply before sending frames over a network.
        # Only applied for local webcams (CAMERA_SOURCE == 0) — RTSP/HTTP streams
        # are already compressed by the camera itself.
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
        if CAMERA_SOURCE == 0:
            _, encimg = cv2.imencode(".jpg", frame, encode_param)
            frame = cv2.imdecode(encimg, 1)

        # Mirror horizontally for local webcams only.
        # CCTV streams are already physically oriented correctly.
        if CAMERA_SOURCE == 0:
            frame = cv2.flip(frame, 1)

        # ------------------------------------------------------------------
        # 1. Detection
        # ------------------------------------------------------------------
        person_count, person_boxes = detect_people(frame)
        face_count,   face_boxes   = detect_faces(frame)
        appliance_on, brightness   = detect_appliance(frame)

        # Combined occupancy: person body OR a visible face counts as occupied
        occupied = (person_count > 0) or (face_count > 0)
        effective_count = 1 if occupied else 0

        # ------------------------------------------------------------------
        # 2. Privacy anonymisation
        #    (must happen BEFORE encoding the frame for the stream)
        # ------------------------------------------------------------------
        frame = blur_people(frame, person_boxes)
        frame = pixelate_faces(frame, face_boxes)

        # ------------------------------------------------------------------
        # 3. Logic engine + energy savings
        # ------------------------------------------------------------------
        waste_detected = _logic.update(effective_count, appliance_on)
        energy_savings = _logic.get_energy_savings()
        latency        = time.time() - start_time

        # ------------------------------------------------------------------
        # 4. Metrics (auto ground-truth: waste=1 when empty+appliance on)
        # ------------------------------------------------------------------
        ground_truth = 1 if (not occupied and appliance_on) else 0
        _evaluator.update(ground_truth, int(waste_detected))
        metrics = _evaluator.compute()

        # ------------------------------------------------------------------
        # 5. Update shared API state
        # ------------------------------------------------------------------
        shared.update_state({
            "people":        person_count,
            "faces":         face_count,
            "occupied":      occupied,
            "appliance_on":  appliance_on,
            "waste_detected": waste_detected,
            "brightness":    int(brightness),
            "latency":       latency,
        })
        shared.update_metrics({
            "precision":          metrics["Precision"],
            "recall":             metrics["Recall"],
            "f1_score":           metrics["F1 Score"],
            "false_trigger_rate": metrics["False Trigger Rate"],
        })
        shared.update_energy(energy_savings)

        # ------------------------------------------------------------------
        # 6. MQTT device control (publish only on state change)
        # ------------------------------------------------------------------
        light_state = "OFF" if waste_detected else "ON"
        fan_state   = "OFF" if waste_detected else "ON"

        if light_state != prev_light_state:
            _mqtt.publish_command(f"wattwatch/{MQTT_ROOM}/lights/cmd", light_state)
            prev_light_state = light_state

        if fan_state != prev_fan_state:
            _mqtt.publish_command(f"wattwatch/{MQTT_ROOM}/fan/cmd", fan_state)
            prev_fan_state = fan_state

        # Also publish the overall waste status for any dashboard subscribers
        _mqtt.publish_waste(waste_detected)

        # ------------------------------------------------------------------
        # 7. Encode anonymised frame for /stream endpoint
        # ------------------------------------------------------------------
        _draw_boxes(frame, person_boxes)
        _draw_overlay(frame, {
            "people":        person_count,
            "faces":         face_count,
            "appliance_on":  appliance_on,
            "waste_detected": waste_detected,
            "brightness":    int(brightness),
            "latency":       latency,
        }, metrics)

        _, jpeg = cv2.imencode(".jpg", frame, encode_param)
        shared.set_frame(bytes(jpeg))

        # ------------------------------------------------------------------
        # 8. Optional local display window
        # ------------------------------------------------------------------
        if not HEADLESS:
            cv2.imshow("Watt-Watch | Privacy Mode", frame)
            if cv2.waitKey(1) == 27:   # ESC to quit
                break

        # Console progress line (overwrites itself)
        print(
            f"\rPeople: {person_count} | "
            f"Waste: {'YES' if waste_detected else 'NO'} | "
            f"F1: {metrics['F1 Score']:.2f} | "
            f"Latency: {latency:.2f}s",
            end="",
            flush=True,
        )

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    _mqtt.disconnect()
    print("\n✅ CV loop stopped.")


if __name__ == "__main__":
    main()