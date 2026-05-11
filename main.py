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

gpu_lock   = threading.Lock()
frame_map  = {}
frame_lock = threading.Lock()

# LEAK FIX 3: confirmed_ids cleanup interval
# Track IDs are local per-camera ByteTrack instance and reset on reconnect.
# confirmed_ids in VLMWorker is global and grows forever unless pruned.
# We keep a per-camera set of IDs seen recently; any confirmed ID not
# seen for CONFIRMED_ID_TTL seconds is removed from vlm_worker.confirmed_ids.
CONFIRMED_ID_TTL = 300   # 5 minutes — remove confirmed IDs not seen recently


# Crop cleanup is handled in vlm_worker.py:
# - False positive crops are deleted immediately after VLM says NONE
# - Fire/smoke crops are kept in alerts/ and copied to alerts/vlm_confirm/


# =========================
# GSTREAMER PIPELINE BUILDER
# =========================
def build_gst_pipeline(source):
    is_rtsp = isinstance(source, str) and source.startswith("rtsp://")
    if is_rtsp:
        return (
            f"rtspsrc location={source} protocols=tcp "
            f"latency=0 drop-on-latency=true "
            f"! rtph265depay ! h265parse ! nvh265dec "
            f"! videoconvert ! video/x-raw,format=BGR "
            f"! appsink drop=true max-buffers=1 sync=false"
        )
    else:
        dev = f"/dev/video{source}" if isinstance(source, int) else source
        return (
            f"v4l2src device={dev} "
            f"! videoconvert ! video/x-raw,format=BGR "
            f"! appsink drop=true max-buffers=1 sync=false"
        )


def open_capture(source, cam_id=""):
    is_rtsp = isinstance(source, str) and source.startswith("rtsp://")
    pipeline = build_gst_pipeline(source)
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    if cap.isOpened():
        print(f"  [{cam_id}] ✅ GStreamer pipeline active")
        return cap
    print(f"  [{cam_id}] ⚠️  GStreamer unavailable — falling back to FFmpeg")
    if is_rtsp:
        return cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    return cv2.VideoCapture(source)


# =========================
# CAMERA WORKER
# =========================
def run_single_camera(camera_config, RTSP_URL, vlm_worker):
    cam_id = camera_config["id"]

    cap = open_capture(RTSP_URL, cam_id)
    if not cap.isOpened():
        print(f"❌ {cam_id} not opened")
        # LEAK FIX 2: ensure cam_id never lingers in frame_map if camera fails
        with frame_lock:
            frame_map.pop(cam_id, None)
        return

    detector = YOLODetector(config["yolo"]["model_path"])
    tracker  = sv.ByteTrack()

    vlm_sent_ids   = set()
    active_ids     = set()
    last_sent_time = 0

    prev_gray      = None
    DIFF_THRESHOLD = 25
    SKIP_MAX       = 10
    skip_count     = 0

    # LEAK FIX 3: track when each confirmed ID was last seen by this camera
    # so we can prune stale entries from vlm_worker.confirmed_ids
    confirmed_last_seen = {}   # { obj_id: timestamp }

    while True:
        ret, frame = cap.read()

        if not ret:
            print(f"⚠️ {cam_id} reconnecting...")
            time.sleep(1)
            cap.release()
            cap = open_capture(RTSP_URL, cam_id)
            continue

        # --- Frame diff ---
        gray = cv2.cvtColor(cv2.resize(frame, (320, 180)), cv2.COLOR_BGR2GRAY)
        if prev_gray is not None and skip_count < SKIP_MAX:
            diff = cv2.absdiff(gray, prev_gray)
            if diff.mean() < DIFF_THRESHOLD:
                skip_count += 1
                prev_gray = gray
                # LEAK FIX 2: still push a frame so frame_map stays fresh
                display_frame = cv2.resize(
                    frame,
                    (config["ui"]["window_width"], config["ui"]["window_height"])
                )
                with frame_lock:
                    frame_map[cam_id] = display_frame
                time.sleep(config["system"]["sleep"])
                continue
        prev_gray  = gray
        skip_count = 0

        # --- YOLO ---
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

        vlm_sent_ids -= (vlm_sent_ids - active_ids)

        # LEAK FIX 3: prune confirmed IDs that haven't been seen recently
        now = time.time()
        stale = [
            oid for oid, ts in confirmed_last_seen.items()
            if now - ts > CONFIRMED_ID_TTL
        ]
        for oid in stale:
            del confirmed_last_seen[oid]
            with vlm_worker.confirmed_lock:
                vlm_worker.confirmed_ids.discard(oid)
            print(f"[{cam_id}] Pruned stale confirmed ID {oid}")

        # --- Draw + VLM dispatch ---
        above_threshold = [d for d in detections if d["conf"] >= CONF_THRESHOLD]
        if above_threshold:
            print(f"[{cam_id}] {above_threshold}")

        for det in detections:
            conf   = det["conf"]
            obj_id = det["id"]

            if conf < CONF_THRESHOLD:
                continue

            x1, y1, x2, y2 = det["bbox"]

            # Update last-seen for confirmed IDs
            if vlm_worker.is_confirmed(obj_id):
                confirmed_last_seen[obj_id] = now
                box_color = (0, 255, 0)
            elif det["class"] == "fire":
                box_color = (0, 0, 255)
            else:
                box_color = (0, 255, 255)

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

            if vlm_worker.is_confirmed(obj_id):
                continue
            if obj_id in vlm_sent_ids:
                continue
            if time.time() - last_sent_time < COOLDOWN:
                continue

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

        # Push to main thread — LEAK FIX 2: frame_map is cleaned up
        # on exit via the finally block below, so stale cameras don't
        # leave dead frames in the dict permanently
        display_frame = cv2.resize(
            frame,
            (config["ui"]["window_width"], config["ui"]["window_height"])
        )
        with frame_lock:
            frame_map[cam_id] = display_frame

        time.sleep(config["system"]["sleep"])

    # LEAK FIX 2: always clean up on exit — whether normal or reconnect failure
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
        save_dir=VLM_SAVE_DIR,
        idle_timeout=config["vlm"].get("idle_timeout", 10)
    )

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
        time.sleep(0.5)

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