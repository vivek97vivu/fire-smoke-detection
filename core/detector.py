from ultralytics import YOLO
import torch

class YOLODetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

        # ✅ Move once to GPU
        self.model.to("cuda")

        # ✅ Force FP32 (fix dtype issue)
        self.model.model.float()

        # ✅ Disable fusion completely (CRITICAL FIX)
        self.model.model.fuse = lambda *args, **kwargs: self.model.model

    def detect(self, frame):
        # ⚡ Use __call__ directly (avoid re-setup)
        results = self.model(
            frame,
            conf=0.7,
            device=0,
            verbose=False
        )

        detections = []

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                detections.append({
                    "class": self.model.names[cls],
                    "conf": conf,
                    "bbox": (x1, y1, x2, y2)
                })

        return detections