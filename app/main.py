# app/main.py
import cv2
import time
from app.cv.camera import get_camera
from app.cv.detector import detect_people
from app.cv.appliance import detect_appliance
from app.cv.privacy import blur_people
from app.logic.engine import WasteDetector
from app.mqtt.client import MQTTClient
from app.config import FRAME_WIDTH, FRAME_HEIGHT
from app.config import JPEG_QUALITY
from app.cv.face_detector import detect_faces
from app.metrics.evaluator import Evaluator 
from app.api.state import latest_state,latest_metrics

# Initialize logic engine with delay (seconds)
logic = WasteDetector(delay_seconds=5)
mqtt=MQTTClient()
evaluator=Evaluator()

def draw_boxes(frame, boxes):
    """Draw bounding boxes around detected people."""
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)


def main():
    cap = get_camera()
    prev_light_state = None
    prev_fan_state = None

    while True:
        ret, frame = cap.read()
        start_time = time.time()
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
        _, encimg = cv2.imencode('.jpg', frame, encode_param)
        frame = cv2.imdecode(encimg, 1)
        if not ret:
            break

        # Mirror view
        frame = cv2.flip(frame, 1)

        # ===== Detection =====
        person_count, boxes = detect_people(frame)
        appliance_on, brightness = detect_appliance(frame)
        face_count, face_boxes = detect_faces(frame)
        
        ## ====Occupancy Logic====
        occupied = (person_count > 0) or (face_count > 0)
        effective_person_count = 1 if occupied else 0    
        
        # ===== Privacy =====
        frame = blur_people(frame, boxes)

        # ===== Logic Engine =====
        waste_detected = logic.update(effective_person_count, appliance_on)
        latency = time.time() - start_time
        
        # ===== Auto Ground Truth for Real-Time Metrics =====
        # Define expected correct behavior
        ground_truth = 1 if (not occupied and appliance_on) else 0

        # Update evaluator every frame
        evaluator.update(ground_truth, int(waste_detected))

        # Compute metrics
        metrics = evaluator.compute()
        
        # Update shared state for API
        latest_state.update({
            "people": person_count,
            "faces": face_count,
            "occupied": occupied,
            "appliance_on": appliance_on,
            "waste_detected": waste_detected,
            "brightness": int(brightness),
            "latency": latency
        })

        latest_metrics.update({
            "precision": metrics["Precision"],
            "recall": metrics["Recall"],
            "f1_score": metrics["F1 Score"],
            "false_trigger_rate": metrics["False Trigger Rate"]
        })
        print(
            f"\rF1: {metrics['F1 Score']:.2f} | "
            f"Precision: {metrics['Precision']:.2f} | "
            f"Recall: {metrics['Recall']:.2f} | "
            f"False Trig: {metrics['False Trigger Rate']:.2f}",
            end=""
        )

        # ===== Device Control with State Tracking =====

        light_state = "OFF" if waste_detected else "ON"
        fan_state   = "OFF" if waste_detected else "ON"

        # Publish only when state changes
        if light_state != prev_light_state:
            mqtt.publish_command("wattwatch/room101/lights/cmd", light_state)
            prev_light_state = light_state

        if fan_state != prev_fan_state:
            mqtt.publish_command("wattwatch/room101/fan/cmd", fan_state)
            prev_fan_state = fan_state

        # Draw boxes
        draw_boxes(frame, boxes)

        # ===== Display Info =====
        cv2.putText(frame, f"People: {person_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        cv2.putText(frame, f"Brightness: {int(brightness)}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        cv2.putText(frame, f"Appliance: {'ON' if appliance_on else 'OFF'}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        
        cv2.putText(frame, f"Latency: {latency:.2f}s", (10, 220),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
        
        cv2.putText(frame, f"Light ROI Active: {appliance_on}", (10, 160),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
        
        cv2.putText(frame, f"F1: {metrics['F1 Score']:.2f}", (10, 190),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
        
        cv2.putText(frame, f"Faces: {face_count}", (10, 120),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

        # Waste status
        status_text = "WASTE DETECTED" if waste_detected else "SECURE"
        status_color = (0, 0, 255) if waste_detected else (0, 255, 0)

        cv2.putText(frame, status_text, (10, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 2)

        cv2.imshow("Watt-Watch | Stage 4", frame)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()