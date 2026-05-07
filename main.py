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


# =========================
# GSTREAMER PIPELINE BUILDER
# =========================
def build_gst_pipeline(source):
    """
    Builds a low-latency GStreamer pipeline.

    RTSP cameras (H265 / HEVC):
      rtspsrc           — pulls RTSP stream over TCP
      drop-on-latency   — drops old frames instead of buffering them
      latency=0         — no jitter buffer (safe for LAN cameras)
      rtph265depay      — strips RTP wrapper
      h265parse         — parses NAL units
      nvh265dec         — GPU hardware decode on RTX 4080 (zero CPU cost)
      videoconvert      — converts to BGR for OpenCV
      appsink           — max-buffers=1 + drop=true = always the latest frame

    Webcam (int index or /dev/videoX):
      v4l2src → videoconvert → appsink
    """
    is_rtsp = isinstance(source, str) and source.startswith("rtsp://")

    if is_rtsp:
        return (
            f"rtspsrc location={source} protocols=tcp "
            f"latency=0 drop-on-latency=true "
            f"! rtph265depay "
            f"! h265parse "
            f"! nvh265dec "
            f"! videoconvert "
            f"! video/x-raw,format=BGR "
            f"! appsink drop=true max-buffers=1 sync=false"
        )
    else:
        dev = f"/dev/video{source}" if isinstance(source, int) else source
        return (
            f"v4l2src device={dev} "
            f"! videoconvert "
            f"! video/x-raw,format=BGR "
            f"! appsink drop=true max-buffers=1 sync=false"
        )


def open_capture(source, cam_id=""):
    """
    Try GStreamer first. Silently fall back to FFmpeg/default if unavailable.
    Returns an opened cv2.VideoCapture.
    """
    is_rtsp = isinstance(source, str) and source.startswith("rtsp://")

    # Try GStreamer
    pipeline = build_gst_pipeline(source)
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    if cap.isOpened():
        print(f"  [{cam_id}] ✅ GStreamer pipeline active (hardware decode)")
        return cap

    print(f"  [{cam_id}] ⚠️  GStreamer unavailable — falling back to FFmpeg")
    if is_rtsp:
        return cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    else:
        return cv2.VideoCapture(source)


# =========================
# CAMERA WORKER
# =========================
def run_single_camera(camera_config, RTSP_URL, vlm_worker):
    cam_id = camera_config["id"]

    cap = open_capture(RTSP_URL, cam_id)
    if not cap.isOpened():
        print(f"❌ {cam_id} not opened")
        return

    detector = YOLODetector(config["yolo"]["model_path"])
    tracker  = sv.ByteTrack()

    vlm_sent_ids   = set()
    active_ids     = set()
    last_sent_time = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            print(f"⚠️ {cam_id} reconnecting...")
            time.sleep(1)
            cap.release()
            cap = open_capture(RTSP_URL, cam_id)   # reconnect also uses GStreamer
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

        # Free sent-but-not-confirmed IDs that left the frame
        vlm_sent_ids -= (vlm_sent_ids - active_ids)

        # --- Draw + VLM dispatch ---
        if detections:
            print(f"[{cam_id}] {detections}")

        for det in detections:
            conf   = det["conf"]
            obj_id = det["id"]

            if conf < CONF_THRESHOLD:
                continue

            x1, y1, x2, y2 = det["bbox"]

            if vlm_worker.is_confirmed(obj_id):
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

            # GATE 1: already confirmed by VLM — never send again
            if vlm_worker.is_confirmed(obj_id):
                continue

            # GATE 2: already queued this appearance
            if obj_id in vlm_sent_ids:
                continue

            # GATE 3: global cooldown
            if time.time() - last_sent_time < COOLDOWN:
                continue

            # GATE 4: VLM not loaded yet
            if vlm_worker.vlm is None:
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

        # Push to main thread for display
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