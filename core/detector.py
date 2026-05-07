from ultralytics import YOLO


class YOLODetector:
    def __init__(self, model_path):
        self.model_path = model_path
        self.is_engine  = model_path.endswith(".engine")

        if self.is_engine:
            # TensorRT engine — already compiled for GPU.
            # Do NOT call .to(), .float(), or patch .fuse — these are
            # PyTorch-only operations that crash on exported formats.
            # Device and precision are baked into the engine at export time.
            self.model = YOLO(model_path, task="detect")
            print(f"⚡ TensorRT engine loaded: {model_path}")
        else:
            # PyTorch .pt model — move to GPU and fix dtype/fusion
            self.model = YOLO(model_path)
            self.model.to("cuda")
            self.model.model.float()
            self.model.model.fuse = lambda *args, **kwargs: self.model.model
            print(f"🔥 PyTorch model loaded: {model_path}")

    def detect(self, frame, conf=0.7):
        if self.is_engine:
            # For TensorRT: pass device explicitly in predict call
            results = self.model.predict(
                frame,
                conf=conf,
                device=0,
                verbose=False
            )
        else:
            results = self.model(
                frame,
                conf=conf,
                device=0,
                verbose=False
            )

        detections = []
        for r in results:
            for box in r.boxes:
                cls  = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                detections.append({
                    "class": self.model.names[cls],
                    "conf":  conf,
                    "bbox":  (x1, y1, x2, y2)
                })

        return detections