import threading
import os
import shutil
from queue import Queue
import time

class VLMWorker:
    def __init__(self, vlm_class, model_path, decision, save_dir):
        self.queue = Queue(maxsize=50)
        self.decision = decision
        self.vlm = None
        self.save_dir = save_dir
        self.last_result = None
        self.last_time = 0

        t = threading.Thread(
            target=self.load_model,
            args=(vlm_class, model_path),
            daemon=True
        )
        t.start()

        t2 = threading.Thread(target=self.run, daemon=True)
        t2.start()

    def load_model(self, vlm_class, model_path):
        print("🚀 Loading VLM in background...")
        self.vlm = vlm_class(model_path) 
        print("✅ VLM READY")

    def add_task(self, img_path, detections):
        if self.vlm is None:
            return

        if not self.queue.full():
            self.queue.put((img_path, detections))

    def run(self):
        while True:
            if self.vlm is None:
                continue

            img_path, detections = self.queue.get()

            result = self.vlm.verify(img_path)

            if result in ["FIRE", "SMOKE"]:
                print(f"🚨 {result} DETECTED (CONFIRMED)")

                self.last_result = result
                self.last_time = time.time()

                # save + alert
                filename = os.path.basename(img_path)
                new_name = f"{result.lower()}_{filename}"
                save_path = os.path.join(self.save_dir, new_name)

                shutil.copy(img_path, save_path)

                self.decision.send_alert(img_path, detections, result)

            else:
                print("❌ Skipped (False detection)")