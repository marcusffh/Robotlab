# aruco_utils.py
import cv2
import numpy as np

# Use a common dictionary (change if your printouts use another)
DICT = cv2.aruco.DICT_6X6_250

def get_aruco():
    aruco = cv2.aruco
    dictionary = aruco.getPredefinedDictionary(DICT)
    params = aruco.DetectorParameters_create()
    return aruco, dictionary, params

def detect_markers(img_bgr):
    """Return (corners, ids) from a BGR frame."""
    aruco, dictionary, params = get_aruco()
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = aruco.detectMarkers(gray, dictionary, parameters=params)
    return corners, ids

def largest_marker_index(corners):
    """Pick the index of the largest detected marker by area."""
    if not corners:
        return None
    areas = [cv2.contourArea(c.reshape(-1,1,2).astype(np.float32)) for c in corners]
    return int(np.argmax(areas))

def side_len_pixels(corners_one):
    """Mean edge length (pixels) for one marker's 4 corners."""
    c = corners_one.reshape(-1, 2)
    d = lambda a, b: np.linalg.norm(c[a] - c[b])
    edges = [d(0,1), d(1,2), d(2,3), d(3,0)]
    return float(np.mean(edges))

def draw_axes_and_box(img, corners, ids, K, dist, marker_len_mm):
    """Draw detected boxes and pose axes (if K provided)."""
    aruco, dictionary, params = get_aruco()
    if ids is None:
        return img
    cv2.aruco.drawDetectedMarkers(img, corners, ids)
    if K is not None and marker_len_mm is not None:
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, marker_len_mm, K, dist if dist is not None else None
        )
        for rvec, tvec in zip(rvecs, tvecs):
            cv2.drawFrameAxes(img, K, dist if dist is not None else np.zeros(5), rvec, tvec, marker_len_mm*0.5)
    return img

def build_camera_matrix(f_px, cx, cy):
    return np.array([[f_px, 0, cx],
                     [0,    f_px, cy],
                     [0,    0,    1 ]], dtype=np.float32)
