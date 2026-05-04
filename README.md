# 🔥 Fire & Smoke Detection System

### Real-Time AI Pipeline using YOLO + Vision-Language Model (VLM)

---

## 🚀 Overview

This project is a **real-time fire & smoke detection system** designed for CCTV / RTSP surveillance.

It combines:

* ⚡ **YOLO** → fast object detection
* 🧠 **Vision-Language Model (Qwen VLM)** → intelligent validation
* 🔁 **Tracking + Filtering** → eliminates false positives & duplicate alerts

👉 The result is a **high-precision, low-noise alert system** suitable for real-world deployment.

---

## 🎯 Key Highlights

* 🔥 Detects **fire and smoke in real-time**
* 🧠 Uses **VLM to reduce false positives**
* ⚠️ Dual-stage alert system:

  * YOLO → *possible detection*
  * VLM → *confirmed alert*
* 🚫 Eliminates duplicate alerts using:

  * Object tracking
  * Cooldown logic
  * Duplicate frame filtering
* ⚙️ Fully configurable via YAML
* 📦 Clean modular architecture (production-ready)

---

## 🧠 Why This Project Matters

Traditional detection systems rely only on object detection → **high false positives**.

This system introduces:

```text
Detection → Verification → Decision
```

✔ Improves reliability
✔ Reduces noise
✔ Makes system usable in real environments

---

## 🏗️ System Architecture

```text
Camera (RTSP / Webcam)
        ↓
YOLO Detection (fire/smoke)
        ↓
Tracking + Filtering
        ↓
Crop Region of Interest
        ↓
VLM Verification (Qwen)
        ↓
🚨 Alert (ONLY if confirmed)
```

---

## 📂 Project Structure

```bash
fire_smoke/
├── core/
│   ├── detector.py        # YOLO inference
│   ├── vlm.py             # VLM verification
│   ├── vlm_worker.py      # async VLM processing
│   ├── decision.py        # alert handling
│   └── tracker.py         # object tracking
│
├── config/
│   └── config.yaml        # central configuration
│
├── alerts/
│   └── vlm_confirm/       # verified outputs
│
├── models/
│   ├── yolo/best.pt
│   └── Qwen3-VL-2B-Instruct/
│
├── main.py
└── README.md
```

---

## ⚙️ Configuration-Driven Design

All system behavior is controlled via `config.yaml`.

### Example:

```yaml
yolo:
  conf_threshold: 0.8

system:
  cooldown: 2
  resize: [224, 224]

vlm:
  device: cuda
  valid_labels: ["FIRE", "SMOKE"]
```

👉 No code changes needed → only config tuning

---

## 🚀 Installation

```bash
git clone https://github.com/vivek97vivu/fire-smoke-detection.git
cd fire-smoke-detection

pip install -r requirements.txt
```

---

## ▶️ Run the System

```bash
python main.py
```

---

## 🎥 Controls

| Key   | Action |
| ----- | ------ |
| `q`   | Quit   |
| `ESC` | Quit   |

---

## 🚨 Alert System

### 🟡 Stage 1 — YOLO Detection

```text
Possible FIRE / SMOKE
```

### 🔴 Stage 2 — VLM Confirmation

```text
FIRE FIRE FIRE 🚨
SMOKE DETECTED ⚠️
```

👉 Alerts are triggered **only after VLM confirmation**

---

## 📸 Output

| Folder                | Description         |
| --------------------- | ------------------- |
| `alerts/`             | Raw YOLO detections |
| `alerts/vlm_confirm/` | Verified fire/smoke |

---

## ⚡ Performance Notes

* YOLO handles real-time detection
* VLM is selectively triggered → optimized usage
* FP16 inference reduces GPU memory
* Tracking prevents redundant processing

---

## 🧪 Engineering Decisions

* **Two-stage pipeline** → improves precision
* **Async VLM worker** → non-blocking inference
* **Tracking-based filtering** → avoids duplicate alerts
* **Config-driven system** → easy deployment & tuning

---

## 🚀 Future Enhancements

* Multi-camera support
* Kafka / REST alert integration
* Alarm system (sound + blinking UI)
* Edge deployment (Jetson / TensorRT)
* Model quantization (reduce GPU usage)

---
