<div align="center">

# 🔥 **Fire & Smoke Detection Engine**

### 🚨 Real-Time Fire & Smoke Detection for Smart Surveillance Systems

A **production-grade AI pipeline** built for **real-time CCTV / RTSP monitoring**, combining **fast detection + intelligent verification** for high-precision alerts.

> ⚙️ Powered by **YOLO TensorRT (Detection)** + **Qwen3-VL (Verification)**
> 🧠 Designed for **low false positives, high reliability deployments**
> 🧩 Part of the **CampNeuron AI Series** — engineered by the **Algosium AI Team**

---

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)](#)
[![CUDA](https://img.shields.io/badge/CUDA-12.x-green?logo=nvidia&logoColor=white)](#)
[![YOLO](https://img.shields.io/badge/YOLO-TensorRT-success?logo=nvidia&logoColor=white)](#)
[![VLM](https://img.shields.io/badge/Qwen3--VL-2B-orange)](#)
[![GStreamer](https://img.shields.io/badge/GStreamer-H265-blue)](#)
[![Platform](https://img.shields.io/badge/Platform-Linux%20|%20x86__64-lightgrey?logo=linux&logoColor=white)](#)

</div>

---

## ⚡ Core Stack

| Component | Purpose |
|---|---|
| 🔥 **YOLO TensorRT Engine** | Real-time fire & smoke detection (hardware accelerated) |
| 🧠 **Qwen3-VL (2B)** | Visual verification — eliminates false positives |
| 🎥 **GStreamer H265 Pipeline** | Low-latency RTSP decode via GPU (nvh265dec) |
| 🔁 **ByteTrack Object Tracking** | Per-object deduplication across frames |
| 💤 **Lazy VLM Loading** | VLM loads on detection, unloads after idle — saves ~2.5GB VRAM |
| 🧵 **Multi-Camera Threading** | Up to 25 cameras, GPU-serialized, Qt-safe display |
| ⚙️ **YAML Config Engine** | Fully configurable — no code changes needed |

---

## 🚀 Pipeline Overview

```text
Camera (RTSP H265 / Webcam)
        ↓
GStreamer Hardware Decode (nvh265dec)
        ↓
Frame Differencing (skip static frames)
        ↓
YOLO TensorRT Detection (Fire / Smoke)
        ↓
ByteTrack — assign stable object IDs
        ↓
Per-ID Filtering (each ID → VLM once only)
        ↓
VLM Verification (Qwen3-VL) ← lazy loaded
        ↓
🚨 ALERT + Save (only if confirmed)
```

---

## 🎯 Key Features

* 🔥 Real-time fire & smoke detection via TensorRT engine
* 🧠 VLM verification using Qwen3-VL — strict false positive elimination
* 💤 Lazy VLM load/unload — GPU memory freed when no incidents detected
* 📷 Multi-camera support (up to 25 RTSP streams simultaneously)
* ⚡ GStreamer H265 hardware decode — 30–60ms lower latency vs FFmpeg
* 🔁 Per-object confirmed ID tracking — each fire/smoke object sent to VLM exactly once
* 🎯 Frame differencing — YOLO skipped on static frames (~60% GPU reduction)
* 🧵 GPU-serialized multi-thread architecture — no VRAM contention
* 📁 Automatic alert storage — false positives auto-deleted, confirmed events kept
* 🚫 6 memory/disk/GPU leaks fixed for long-running production stability

---

## 📂 Project Structure

```bash
fire_smoke/
├── core/
│   ├── detector.py        # YOLO TensorRT inference (.pt and .engine)
│   ├── vlm.py             # Qwen3-VL verification with <think> tag parsing
│   ├── vlm_worker.py      # Lazy load/unload async VLM worker
│   ├── decision.py        # Alert cooldown and dispatch
│
├── config/
│   └── config.yaml        # Central configuration (cameras, models, thresholds)
│
├── alerts/
│   └── vlm_confirm/       # VLM-confirmed fire/smoke images (kept permanently)
│
├── models/
│   ├── yolo/best.engine   # TensorRT compiled YOLO model
│   └── Qwen3-VL-2B-Instruct/
│
├── main.py
└── README.md
```

---

## ⚙️ Configuration

All system behavior is controlled via `config.yaml`. No code changes needed.

```yaml
camera:
  - id: cam_1
    source: 0                         # webcam
  - id: cam_2
    source: "rtsp://user:pass@ip/..."  # RTSP H265 stream
  # up to cam_25 supported

yolo:
  model_path: models/yolo/best.engine
  conf_threshold: 0.8

vlm:
  model_path: models/Qwen3-VL-2B-Instruct
  idle_timeout: 10    # seconds idle before VLM unloads from GPU
  device: cuda
  use_fp16: true
  valid_labels: ["FIRE", "SMOKE"]

system:
  cooldown: 2         # seconds between VLM sends per camera
  resize: [160, 160]  # crop size sent to VLM
```

---

## 🚀 Installation

```bash
git clone https://github.com/vivek97vivu/fire-smoke-detection.git
cd fire-smoke-detection

pip install -r requirements.txt
pip install accelerate   # required for VLM device_map
```

### Requirements

* NVIDIA GPU (RTX series recommended)
* CUDA 12.x
* GStreamer with `gstreamer1.0-plugins-bad` (for nvh265dec)
* Python 3.12

```bash
# Verify GStreamer H265 support
gst-inspect-1.0 nvh265dec
```

---

## ▶️ Run

```bash
python main.py
```

---

## 🎥 Controls

| Key | Action |
|---|---|
| `q` | Quit all cameras |
| `ESC` | Quit all cameras |

---

## 🚨 Alert System

### Stage 1 — YOLO Detection
```
Bounding box drawn — object assigned stable track ID
```

### Stage 2 — VLM Verification (lazy loaded)
```
FIRE FIRE FIRE 🚨   →  alerts/vlm_confirm/fire_<id>_<ts>.jpg
SMOKE DETECTED ⚠️  →  alerts/vlm_confirm/smoke_<id>_<ts>.jpg
False positive      →  crop deleted automatically
```

Alerts trigger **only after VLM confirmation**. Each track ID is verified **exactly once** per appearance.

---

## 📸 Output

| Folder | Contents |
|---|---|
| `alerts/` | Raw crops sent to VLM (false positives auto-deleted) |
| `alerts/vlm_confirm/` | VLM-confirmed fire/smoke images (kept permanently) |

---

## ⚡ Performance

| Metric | Value |
|---|---|
| YOLO latency | ~5–10ms (TensorRT FP16) |
| GStreamer decode latency | ~10–20ms (H265 GPU) vs ~60ms FFmpeg |
| VLM inference | ~800ms–1.5s (Qwen3-2B FP16) |
| GPU memory (idle, VLM unloaded) | ~2.1 GB |
| GPU memory (VLM active) | ~4.8 GB |
| Cameras supported | Up to 25 (GPU-serialized) |

---

## 🧪 Engineering Decisions

| Decision | Reason |
|---|---|
| Two-stage YOLO + VLM | YOLO is fast but imprecise; VLM eliminates false positives |
| Lazy VLM load/unload | Fires are rare — no reason to keep 2.5GB loaded 24/7 |
| GStreamer over FFmpeg | Hardware H265 decode, lower latency, drop=true policy |
| GPU lock per camera | Prevents CUDA kernel contention across threads |
| Main-thread display | Qt requires all imshow/waitKey on the main thread |
| Per-ID confirmed set | Prevents the same object triggering repeated VLM calls |
| Frame differencing | 60% YOLO GPU reduction on static CCTV scenes |

---

## 🔮 Future Enhancements

* Kafka / REST webhook alert integration
* Sound alarm + blinking UI on confirmation
* Edge deployment optimization (Jetson Orin)
* 4-bit VLM quantization (bitsandbytes) for further VRAM reduction
* Web dashboard for multi-camera monitoring
* Alert log database (SQLite / PostgreSQL)

---

<div align="center">
Engineered by the <b>Algosium AI Team</b> · CampNeuron AI Series
</div>