import threading
import os
import shutil
import time
import torch
from queue import Queue, Empty


class VLMWorker:
    """
    Lazy-loading VLM worker.
    Loads on first YOLO detection, unloads after idle_timeout seconds.
    Both model AND processor are deleted on unload to fully free GPU/RAM.
    """

    STATE_UNLOADED  = "UNLOADED"
    STATE_LOADING   = "LOADING"
    STATE_READY     = "READY"
    STATE_UNLOADING = "UNLOADING"

    def __init__(self, vlm_class, model_config, decision, save_dir,
                 idle_timeout=10):
        self.vlm_class    = vlm_class
        self.model_config = model_config
        self.decision     = decision
        self.save_dir     = save_dir
        self.idle_timeout = idle_timeout

        self.queue = Queue(maxsize=50)
        self.vlm   = None
        self.state = self.STATE_UNLOADED

        self.last_result    = None
        self.last_time      = 0
        self.last_task_time = 0

        self.confirmed_ids  = set()
        self.confirmed_lock = threading.Lock()
        self._lock          = threading.Lock()

        os.makedirs(self.save_dir, exist_ok=True)

        threading.Thread(target=self._run, daemon=True).start()

    def is_confirmed(self, obj_id):
        with self.confirmed_lock:
            return obj_id in self.confirmed_ids

    def add_task(self, img_path, detections):
        self.last_task_time = time.time()
        if self.queue.full():
            print("[VLM] Queue full — task dropped")
            return
        self.queue.put((img_path, detections))

    def _run(self):
        while True:
            try:
                img_path, detections = self.queue.get(timeout=0.5)
            except Empty:
                self._maybe_unload()
                continue

            if self.vlm is None:
                self._load_vlm()

            obj_id = detections[0].get("id", "x")

            # LEAK FIX 6: catch exceptions explicitly so task is never
            # silently dropped — log the error and continue the loop
            try:
                result = self.vlm.verify(img_path)
            except Exception as e:
                print(f"❌ [VLM] Inference error on task (ID {obj_id}): {e}")
                result = "NONE"

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if result in ("FIRE", "SMOKE"):
                print(f"🚨 {result} CONFIRMED (ID {obj_id})")

                with self.confirmed_lock:
                    self.confirmed_ids.add(obj_id)

                self.last_result = result
                self.last_time   = time.time()

                timestamp = int(time.time() * 1000)
                new_name  = f"{result.lower()}_{obj_id}_{timestamp}.jpg"
                save_path = os.path.join(self.save_dir, new_name)
                shutil.copy(img_path, save_path)
                print(f"✅ Saved: {save_path}")

                self.decision.send_alert(img_path, detections, result)

            else:
                print(f"❌ False positive (ID {obj_id}) — deleting crop")
                # Delete the crop only on false positive — no need to keep it.
                # Confirmed crops are already saved to vlm_confirm/ above.
                try:
                    if os.path.exists(img_path):
                        os.remove(img_path)
                except Exception as e:
                    print(f"[VLM] Could not delete crop: {e}")

    def _maybe_unload(self):
        if self.vlm is None or self.state == self.STATE_LOADING:
            return
        if time.time() - self.last_task_time >= self.idle_timeout:
            self._unload_vlm()

    def _load_vlm(self):
        with self._lock:
            if self.vlm is not None:
                return
            print("🚀 [VLM] Loading model...")
            self.state = self.STATE_LOADING
            self.vlm   = self.vlm_class(self.model_config)
            self.state = self.STATE_READY
            print("✅ [VLM] Ready")

    def _unload_vlm(self):
        with self._lock:
            if self.vlm is None:
                return
            print(f"💤 [VLM] Idle {self.idle_timeout}s — unloading...")
            self.state = self.STATE_UNLOADING

            # LEAK FIX 5: delete both model AND processor
            # Previously only model was deleted — processor kept ~200MB
            # of tokenizer tensors and image embeddings in GPU memory
            del self.vlm.model
            del self.vlm.processor
            del self.vlm
            self.vlm = None

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            self.state = self.STATE_UNLOADED
            print("✅ [VLM] Unloaded — GPU memory released")