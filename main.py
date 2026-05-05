import cv2
import time
import os
import warnings
import numpy as np
import threading

warnings.filterwarnings("ignore")

from core.detector import YOLODetector
from core.vlm import VLMVerifier
from core.decision import DecisionEngine
from core.vlm_worker import VLMWorker
from config.config_loader import load_config
import supervision as sv

# =========================
# LOAD CONFIG
# =========================
config = load_config()

camera_list    = config["camera"]
CONF_THRESHOLD = config["yolo"]["conf_threshold"]
COOLDOWN       = config["system"]["cooldown"]
SAVE_DIR       = config["alerts"]["save_dir"]
VLM_SAVE_DIR   = config["alerts"]["vlm_confirm_dir"]

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(VLM_SAVE_DIR, exist_ok=True)

# gpu_lock  — only one camera runs YOLO at a time (prevents GPU contention)
# frame_map — camera threads write frames here; main thread calls imshow
gpu_lock   = threading.Lock()
frame_map  = {}
frame_lock = threading.Lock()


# =========================
# CAMERA WORKER
# =========================
def run_single_camera(camera_config, RTSP_URL, vlm_worker):
    cam_id = camera_config["id"]

    cap = cv2.VideoCapture(RTSP_URL)
    if not cap.isOpened():
        print(f"❌ {cam_id} not opened")
        return

    detector = YOLODetector(config["yolo"]["model_path"])
    tracker  = sv.ByteTrack()

    # vlm_sent_ids:
    #   IDs queued to VLM during the CURRENT appearance in frame.
    #   Cleared when the object vanishes, so a false-positive object
    #   gets one more VLM check if YOLO picks it up again later.
    #
    # vlm_worker.confirmed_ids (in VLMWorker):
    #   IDs permanently confirmed as FIRE/SMOKE by VLM.
    #   Never cleared — these are NEVER sent to VLM again.
    vlm_sent_ids   = set()
    active_ids     = set()
    last_sent_time = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            print(f"⚠️ {cam_id} reconnecting...")
            time.sleep(1)
            cap.release()
            cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
            continue

        # --- YOLO (GPU serialized) ---
        with gpu_lock:
            detections = detector.detect(frame)

        # --- Tracking ---
        active_ids.clear()

        if detections:
            boxes, scores, class_ids = [], [], []
            for det in detections:
                x1, y1, x2, y2 = det["bbox"]
                boxes.append([x1, y1, x2, y2])
                scores.append(det["conf"])
                class_ids.append(0 if det["class"] == "fire" else 1)

            detections_sv = sv.Detections(
                xyxy=np.array(boxes,      dtype=np.float32),
                confidence=np.array(scores,    dtype=np.float32),
                class_id=np.array(class_ids, dtype=np.int32)
            )
            detections_sv = tracker.update_with_detections(detections_sv)

            new_detections = []
            for det, track_id in zip(detections, detections_sv.tracker_id):
                if track_id is None:
                    continue
                tid = int(track_id)
                active_ids.add(tid)
                new_detections.append({**det, "id": tid})
            detections = new_detections
        else:
            detections = []

        # Free sent-but-not-confirmed IDs that left the frame.
        # This allows a false-positive object one more VLM attempt
        # if it reappears, without spamming VLM while it's visible.
        vanished = vlm_sent_ids - active_ids
        vlm_sent_ids -= vanished

        # --- Draw + VLM dispatch ---
        if detections:
            print(f"[{cam_id}] {detections}")

        for det in detections:
            conf   = det["conf"]
            obj_id = det["id"]

            if conf < CONF_THRESHOLD:
                continue

            x1, y1, x2, y2 = det["bbox"]

            # Choose colour: green border if already VLM-confirmed
            if vlm_worker.is_confirmed(obj_id):
                box_color = (0, 255, 0)          # green = confirmed
            elif det["class"] == "fire":
                box_color = (0, 0, 255)          # red = YOLO only
            else:
                box_color = (0, 255, 255)        # yellow = YOLO only

            cv2.rectangle(frame, (x1, y1), (x2, y2),
                          box_color, config["ui"]["bbox_thickness"])

            if config["ui"]["show_label"]:
                status = "CONFIRMED" if vlm_worker.is_confirmed(obj_id) else f"{conf:.2f}"
                cv2.putText(frame,
                            f"{det['class']} {status} ID:{obj_id}",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            config["ui"]["font_scale"],
                            (0, 255, 0),
                            config["ui"]["font_thickness"])

            if config["ui"]["show_warning"]:
                cv2.putText(frame, det["class"].upper(), (x1, y1 - 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            config["ui"]["font_scale"],
                            (0, 255, 255),
                            config["ui"]["font_thickness"])

            # ── GATE 1: already confirmed by VLM → never send again ──
            if vlm_worker.is_confirmed(obj_id):
                continue

            # ── GATE 2: already queued to VLM this appearance → wait ──
            if obj_id in vlm_sent_ids:
                continue

            # ── GATE 3: global cooldown between sends ──
            if time.time() - last_sent_time < COOLDOWN:
                continue

            # ── GATE 4: VLM not loaded yet ──
            if vlm_worker.vlm is None:
                continue

            # --- Crop and send to VLM ---
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            resize_w, resize_h = config["system"]["resize"]
            crop = cv2.resize(crop, (resize_w, resize_h))

            timestamp = int(time.time() * 1000)
            img_path  = os.path.join(SAVE_DIR, f"crop_{cam_id}_{timestamp}.jpg")
            cv2.imwrite(img_path, crop)

            vlm_sent_ids.add(obj_id)
            last_sent_time = time.time()

            print(f"[INFO] [{cam_id}] Queued for VLM → ID:{obj_id} | {img_path}")
            vlm_worker.add_task(img_path, [det])

        # --- VLM alert overlay ---
        if (
            getattr(vlm_worker, "last_result", None) in ("FIRE", "SMOKE")
            and time.time() - vlm_worker.last_time < 3
        ):
            if vlm_worker.last_result == "FIRE":
                alert_text, alert_color = "FIRE FIRE FIRE", (0, 0, 255)
            else:
                alert_text, alert_color = "SMOKE DETECTED", (0, 165, 255)

            cv2.rectangle(frame, (0, 0), (800, 100), (0, 0, 0), -1)
            cv2.putText(frame, alert_text, (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.0, alert_color, 5)

        # --- Push to main thread for display ---
        display_frame = cv2.resize(
            frame,
            (config["ui"]["window_width"], config["ui"]["window_height"])
        )
        with frame_lock:
            frame_map[cam_id] = display_frame

        time.sleep(config["system"]["sleep"])

    cap.release()
    with frame_lock:
        frame_map.pop(cam_id, None)


# =========================
# MAIN
# =========================
def main():
    print("🚀 Starting Multi-Camera Pipeline...")

    decision = DecisionEngine(cooldown=config["system"]["cooldown"])

    vlm_worker = VLMWorker(
        VLMVerifier,
        config["vlm"],
        decision,
        save_dir=VLM_SAVE_DIR
    )

    # Create all windows on main thread (Qt requirement)
    for camera_config in camera_list:
        win = f"Fire & Smoke Detection - {camera_config['id']}"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win,
                         config["ui"]["window_width"],
                         config["ui"]["window_height"])

    threads = []
    for camera_config in camera_list:
        print(f"🎥 Starting camera: {camera_config['id']}")
        t = threading.Thread(
            target=run_single_camera,
            args=(camera_config, camera_config["source"], vlm_worker),
            daemon=True
        )
        t.start()
        threads.append(t)
        time.sleep(0.5)     # stagger GPU startup

    # --- Display loop on main thread ---
    print("🖥️  Press Q or ESC to quit.")
    while any(t.is_alive() for t in threads):
        with frame_lock:
            current_frames = dict(frame_map)

        for cam_id, frm in current_frames.items():
            cv2.imshow(f"Fire & Smoke Detection - {cam_id}", frm)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            print("🛑 Quitting...")
            break

        time.sleep(0.01)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()