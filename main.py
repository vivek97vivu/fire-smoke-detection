import cv2
import time
import os
import warnings

warnings.filterwarnings("ignore")

from core.detector import YOLODetector
from core.vlm import VLMVerifier
from core.decision import DecisionEngine
from core.vlm_worker import VLMWorker
from config.config_loader import load_config
from core.tracker import SimpleTracker


# =========================
# LOAD CONFIG
# =========================
config = load_config()

# ✅ camera config (list → pick first)
camera_config = config["camera"][0]
RTSP_URL = camera_config["source"]

CONF_THRESHOLD = config["yolo"]["conf_threshold"]
COOLDOWN = config["system"]["cooldown"]

SAVE_DIR = config["alerts"]["save_dir"]
VLM_SAVE_DIR = config["alerts"]["vlm_confirm_dir"]

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(VLM_SAVE_DIR, exist_ok=True)


def main():
    print("🚀 Starting Fire & Smoke Detection Pipeline...")

    cap = cv2.VideoCapture(RTSP_URL)

    if not cap.isOpened():
        print("❌ Camera not opened")
        return

    detector = YOLODetector(config["yolo"]["model_path"])
    decision = DecisionEngine()

    # ✅ tracker inside main
    tracker = SimpleTracker()

    # 🔥 VLM worker
    vlm_worker = VLMWorker(
        VLMVerifier,
        config["vlm"],
        decision,
        save_dir=VLM_SAVE_DIR
    )

    # =========================
    # UI WINDOW SETUP (CONFIG DRIVEN)
    # =========================
    window_name = "Fire & Smoke Detection"

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(
        window_name,
        config["ui"]["window_width"],
        config["ui"]["window_height"]
    )

    last_sent_time = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            print("⚠️ Frame not received. Reconnecting...")
            time.sleep(1)
            cap.release()
            cap = cv2.VideoCapture(RTSP_URL)
            continue

        detections = detector.detect(frame)

        if detections:
            detections = tracker.update(detections)

            print(f"[{camera_config['id']}] {detections}")

            for det in detections:
                conf = det["conf"]
                obj_id = det["id"]

                if conf < CONF_THRESHOLD:
                    continue

                x1, y1, x2, y2 = det["bbox"]

                # =========================
                # COLOR BASED ON CLASS
                # =========================
                color = (0, 0, 255) if det["class"] == "fire" else (0, 255, 255)

                # =========================
                # DRAW BBOX (CONFIG)
                # =========================
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    color,
                    config["ui"]["bbox_thickness"]
                )

                # =========================
                # LABEL (CONFIG)
                # =========================
                if config["ui"]["show_label"]:
                    label = f"{det['class']} {conf:.2f} ID:{obj_id}"

                    cv2.putText(
                        frame,
                        label,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        config["ui"]["font_scale"],
                        (0, 255, 0),
                        config["ui"]["font_thickness"],
                    )

                # =========================
                # WARNING (CONFIG)
                # =========================
                if config["ui"]["show_warning"]:
                    cv2.putText(
                        frame,
                        f"{det['class'].upper()}",
                        (x1, y1 - 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        config["ui"]["font_scale"],
                        (0, 255, 255),
                        config["ui"]["font_thickness"],
                    )

                # =========================
                # COOLDOWN FIRST
                # =========================
                if time.time() - last_sent_time < COOLDOWN:
                    continue

                # =========================
                # TRACKING (ONE ALERT PER OBJECT)
                # =========================
                if not tracker.should_alert(obj_id):
                    continue

                # =========================
                # CHECK VLM READY
                # =========================
                if vlm_worker.vlm is None:
                    continue

                crop = frame[y1:y2, x1:x2]

                if crop.size == 0:
                    continue

                # =========================
                # RESIZE (CONFIG)
                # =========================
                resize_w, resize_h = config["system"]["resize"]
                crop = cv2.resize(crop, (resize_w, resize_h))

                # =========================
                # SAVE IMAGE
                # =========================
                timestamp = int(time.time() * 1000)
                img_path = os.path.join(SAVE_DIR, f"crop_{timestamp}.jpg")

                cv2.imwrite(img_path, crop)

                print(f"[INFO] Sent to VLM (ID {obj_id}): {img_path}")

                # =========================
                # SEND TO VLM
                # =========================
                vlm_worker.add_task(img_path, [det])

                last_sent_time = time.time()


        # =========================
        # 🔥 VLM CONFIRMED ALERT UI (CORRECT PLACE)
        # =========================
        if hasattr(vlm_worker, "last_result"):
            if time.time() - vlm_worker.last_time < 3:

                if vlm_worker.last_result == "FIRE":
                    alert_text = "FIRE FIRE FIRE"
                    color = (0, 0, 255)

                elif vlm_worker.last_result == "SMOKE":
                    alert_text = "SMOKE DETECTED"
                    color = (0, 165, 255)

                cv2.rectangle(frame, (0, 0), (800, 100), (0, 0, 0), -1)

                cv2.putText(
                    frame,
                    alert_text,
                    (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2.0,
                    color,
                    5,
                )

        # =========================
        # DISPLAY (CONFIG RESOLUTION)
        # =========================
        display_frame = cv2.resize(
            frame,
            (
                config["ui"]["window_width"],
                config["ui"]["window_height"]
            )
        )

        cv2.imshow(window_name, display_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == 27:  
            print("🛑 Stopping pipeline...")
            break

        time.sleep(config["system"]["sleep"])

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()