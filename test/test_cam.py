import cv2

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

while True:
    ret, frame = cap.read()
    print("ret:", ret)

    if ret:
        cv2.imshow("Test", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()