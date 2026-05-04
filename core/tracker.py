import math

class SimpleTracker:
    def __init__(self, max_distance=50):
        self.objects = {}   # id → center
        self.next_id = 0
        self.max_distance = max_distance
        self.alerted_ids = set()

    def _get_center(self, bbox):
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    def _distance(self, p1, p2):
        return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

    def update(self, detections):
        tracked = []

        for det in detections:
            center = self._get_center(det["bbox"])

            matched_id = None

            for obj_id, obj_center in self.objects.items():
                if self._distance(center, obj_center) < self.max_distance:
                    matched_id = obj_id
                    break

            if matched_id is None:
                matched_id = self.next_id
                self.next_id += 1

            self.objects[matched_id] = center

            det["id"] = matched_id
            tracked.append(det)

        return tracked

    def should_alert(self, obj_id):
        if obj_id in self.alerted_ids:
            return False

        self.alerted_ids.add(obj_id)
        return True