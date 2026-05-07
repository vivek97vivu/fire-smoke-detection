import cv2

rtsp_url = "rtsp://admin:diffuse123@192.168.0.183:554/cam/realmonitor?channel=1&subtype=0"

pipeline = (
    f"rtspsrc location={rtsp_url} protocols=tcp latency=0 drop-on-latency=true ! "
    "rtph265depay ! h265parse ! nvh265dec ! "
    "videoconvert ! video/x-raw,format=BGR ! "
    "appsink drop=true sync=false max-buffers=1"
)

cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("❌ Cannot open RTSP stream (GStreamer H265)")
    exit()

print("✅ RTSP H265 stream started")

while True:
    ret, frame = cap.read()

    if not ret:
        continue

    cv2.imshow("RTSP - H265 Low Latency", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()