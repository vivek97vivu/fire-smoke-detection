import threading
import os
import shutil
from queue import Queue
import time


class VLMWorker:
    def __init__(self, vlm_class, model_config, decision, save_dir):
        self.queue    = Queue(maxsize=50)
        self.decision = decision
        self.vlm      = None
        self.save_dir = save_dir

        self.last_result = None
        self.last_time   = 0

        # Track IDs that VLM has already confirmed as FIRE or SMOKE.
        # Once an ID is in here it will NEVER be sent to VLM again —
        # regardless of disappearing / reappearing in the frame.
        # Shared across all camera threads (protected by confirmed_lock).
        self.confirmed_ids  = set()
        self.confirmed_lock = threading.Lock()

        os.makedirs(self.save_dir, exist_ok=True)

        threading.Thread(target=self.load_model,
                         args=(vlm_class, model_config),
                         daemon=True).start()

        threading.Thread(target=self.run, daemon=True).start()

    def load_model(self, vlm_class, model_config):
        print("🚀 Loading VLM in background...")
        self.vlm = vlm_class(model_config)
        print("✅ VLM READY")

    def is_confirmed(self, obj_id):
        """Return True if this track ID was already confirmed by VLM."""
        with self.confirmed_lock:
            return obj_id in self.confirmed_ids

    def add_task(self, img_path, detections):
        if self.vlm is None:
            return
        if not self.queue.full():
            self.queue.put((img_path, detections))

    def run(self):
        while True:
            if self.vlm is None:
                time.sleep(0.1)
                continue

            try:
                img_path, detections = self.queue.get(timeout=1)
            except Exception:
                continue

            result = self.vlm.verify(img_path)
            obj_id = detections[0].get("id", "x")

            if result in ("FIRE", "SMOKE"):
                print(f"🚨 {result} CONFIRMED (ID {obj_id})")

                # Permanently mark this track ID as confirmed
                with self.confirmed_lock:
                    self.confirmed_ids.add(obj_id)

                self.last_result = result
                self.last_time   = time.time()

                # Save confirmed image
                timestamp = int(time.time() * 1000)
                new_name  = f"{result.lower()}_{obj_id}_{timestamp}.jpg"
                save_path = os.path.join(self.save_dir, new_name)
                shutil.copy(img_path, save_path)
                print(f"✅ Saved: {save_path}")

                self.decision.send_alert(img_path, detections, result)

            else:
                # VLM said NONE — false positive from YOLO.
                # Do NOT add to confirmed_ids so if YOLO sees it again
                # with higher confidence it gets another VLM check.
                print(f"❌ False positive (ID {obj_id}) — not confirmed")