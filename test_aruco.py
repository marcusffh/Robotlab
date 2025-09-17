# test_aruco_cv2.py
import cv2, numpy as np

aruco = cv2.aruco
DICT = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
params = aruco.DetectorParameters_create()
# (Optional robustness)
params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

while True:
    ok, frame = cap.read()
    if not ok: continue
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = aruco.detectMarkers(gray, DICT, parameters=params)
    if ids is not None:
        aruco.drawDetectedMarkers(frame, corners, ids)
        print("IDs:", ids.flatten().tolist())
    cv2.imshow("view", frame)
    if cv2.waitKey(1) & 0xFF == 27: break  # ESC to quit

cap.release()
cv2.destroyAllWindows()
