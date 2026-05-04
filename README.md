<div align="center">

# 🔥 **Fire & Smoke Detection Engine**

### 🚨 Real-Time Fire & Smoke Detection for Smart Surveillance Systems

A **production-grade AI pipeline** built for **real-time CCTV / RTSP monitoring**, combining **fast detection + intelligent verification** for high-precision alerts.

> ⚙️ Powered by **YOLO (Detection)** + **Qwen VLM (Verification)**
> 🧠 Designed for **low false positives, high reliability deployments**
> 🧩 Part of the **CampNeuron AI Series** — engineered by the **Algosium AI Team**

---

[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python\&logoColor=white)](#)
[![CUDA](https://img.shields.io/badge/CUDA-12.x-green?logo=nvidia\&logoColor=white)](#)
[![YOLO](https://img.shields.io/badge/YOLO-Detection-success?logo=yolo\&logoColor=white)](#)
[![VLM](https://img.shields.io/badge/Qwen-VLM-orange)](#)
[![Platform](https://img.shields.io/badge/Platform-Linux%20|%20x86__64-lightgrey?logo=linux\&logoColor=white)](#)

</div>

---

## ⚡ Core Stack

| Component                 | Purpose                                       |
| ------------------------- | --------------------------------------------- |
| 🔥 **YOLO Model**         | Real-time fire & smoke detection              |
| 🧠 **Qwen3-VL (2B)**      | Visual verification (reduces false positives) |
| 🎥 **OpenCV Pipeline**    | Camera streaming (RTSP / webcam)              |
| 🔁 **Object Tracking**    | Avoid repeated alerts                         |
| ⏱️ **Cooldown System**    | Prevent alert spam                            |
| 🧠 **Async VLM Worker**   | Non-blocking inference                        |
| ⚙️ **YAML Config Engine** | Fully configurable pipeline                   |

---

## 🚀 Pipeline Overview

```text
Camera (RTSP / Webcam)
        ↓
YOLO Detection (Fire / Smoke)
        ↓
Tracking + Filtering
        ↓
Crop Region of Interest
        ↓
VLM Verification (Qwen)
        ↓
🚨 ALERT (Only if confirmed)
```

---

## 🎯 Key Features

* 🔥 Real-time fire & smoke detection
* 🧠 AI verification using Vision-Language Model
* ⚠️ Dual-stage alert system (Detection → Confirmation)
* 🚫 Duplicate alert prevention (tracking + hashing)
* ⚡ Optimized GPU usage (FP16 + selective VLM calls)
* 📁 Automatic alert storage (raw + verified)
* ⚙️ Fully config-driven architecture

---

## 📂 Project Structure

```bash
fire_smoke/
├── core/
│   ├── detector.py
│   ├── vlm.py
│   ├── vlm_worker.py
│   ├── decision.py
│   └── tracker.py
│
├── config/
│   └── config.yaml
│
├── alerts/
│   └── vlm_confirm/
│
├── models/
│   ├── yolo/
│   └── Qwen3-VL-2B-Instruct/
│
├── main.py
└── README.md
```

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
