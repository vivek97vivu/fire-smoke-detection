import cv2

rtsp_url = "rtsp://admin:diffuse123@192.168.0.183:554/cam/realmonitor?channel=1&subtype=0"

cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)

if not cap.isOpened():
    print("❌ Cannot open RTSP stream")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("⚠️ Frame not received")
        continue

    cv2.imshow("Test", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()