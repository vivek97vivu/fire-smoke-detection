import time


class DecisionEngine:
    def __init__(self, cooldown=30):
        self.last_alert_time = 0
        self.cooldown = cooldown  # read from config, not hardcoded

    def should_trigger(self, detections):
        """
        Pre-filter before sending to VLM
        """
        for d in detections:
            if d["conf"] > 0.65:
                return True
        return False

    def final_decision(self, detections, vlm_result):
        """
        Final validation using VLM
        """
        if vlm_result in ["FIRE", "SMOKE"]:
            return True
        return False

    def send_alert(self, image_path, detections, result):
        current_time = time.time()

        # cooldown to avoid alert spam
        if current_time - self.last_alert_time < self.cooldown:
            print(f"⏳ Alert skipped (cooldown {self.cooldown}s)")
            return

        self.last_alert_time = current_time

        alert = {
            "event":      result,
            "image":      image_path,
            "detections": detections,
            "timestamp":  current_time,
        }

        print("\n🚨🚨🚨 FIRE/SMOKE ALERT 🚨🚨🚨")
        print(alert)
        print("=================================\n")