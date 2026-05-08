import threading
import os
import shutil
import time
import torch
from queue import Queue, Empty


class VLMWorker:

    # VLM lifecycle states — readable in UI / logs
    STATE_UNLOADED = "UNLOADED"
    STATE_LOADING  = "LOADING"
    STATE_READY    = "READY"
    STATE_UNLOADING = "UNLOADING"

    def __init__(self, vlm_class, model_config, decision, save_dir,
                 idle_timeout=10):
        self.vlm_class    = vlm_class
        self.model_config = model_config
        self.decision     = decision
        self.save_dir     = save_dir
        self.idle_timeout = idle_timeout   # seconds before unloading

        self.queue = Queue(maxsize=50)
        self.vlm   = None                  # None = unloaded
        self.state = self.STATE_UNLOADED

        self.last_result   = None
        self.last_time     = 0
        self.last_task_time = 0            # updated on every add_task()

        self.confirmed_ids  = set()
        self.confirmed_lock = threading.Lock()

        self._lock = threading.Lock()      # guards vlm load/unload

        os.makedirs(self.save_dir, exist_ok=True)

        # Single worker thread — handles load, inference, unload
        threading.Thread(target=self._run, daemon=True).start()

    # ------------------------------------------------------------------
    # Public API (called from camera threads)
    # ------------------------------------------------------------------

    def is_confirmed(self, obj_id):
        with self.confirmed_lock:
            return obj_id in self.confirmed_ids

    def add_task(self, img_path, detections):
        """
        Queue a crop for VLM verification.
        Also wakes the worker if VLM is currently unloaded.
        """
        self.last_task_time = time.time()

        if self.queue.full():
            print("[VLM] Queue full — task dropped")
            return

        self.queue.put((img_path, detections))

    # ------------------------------------------------------------------
    # Internal worker loop
    # ------------------------------------------------------------------

    def _run(self):
        while True:
            try:
                img_path, detections = self.queue.get(timeout=0.5)
            except Empty:
                # Nothing in queue — check if we should unload
                self._maybe_unload()
                continue

            # Load VLM if not already loaded
            if self.vlm is None:
                self._load_vlm()

            # Run inference
            result = self.vlm.verify(img_path)
            obj_id = detections[0].get("id", "x")

            # Free KV-cache after each inference
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
                print(f"❌ False positive (ID {obj_id})")

    def _maybe_unload(self):
        """Unload VLM if idle for longer than idle_timeout."""
        if self.vlm is None:
            return
        if self.state == self.STATE_LOADING:
            return

        idle_for = time.time() - self.last_task_time
        if idle_for >= self.idle_timeout:
            self._unload_vlm()

    def _load_vlm(self):
        with self._lock:
            if self.vlm is not None:
                return   # another thread already loaded it

            print(f"🚀 [VLM] Loading model (triggered by detection)...")
            self.state = self.STATE_LOADING

            self.vlm   = self.vlm_class(self.model_config)
            self.state = self.STATE_READY

            print(f"✅ [VLM] Ready")

    def _unload_vlm(self):
        with self._lock:
            if self.vlm is None:
                return

            print(f"💤 [VLM] Idle for {self.idle_timeout}s — unloading to free GPU memory...")
            self.state = self.STATE_UNLOADING

            # Delete model and free all GPU memory
            del self.vlm
            self.vlm = None

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            self.state = self.STATE_UNLOADED
            print(f"✅ [VLM] Unloaded — GPU memory released")