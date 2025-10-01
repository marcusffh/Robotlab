# simple_focal_length_height_only.py
# Collect focal-length samples using ONLY the marker's HEIGHT (pixels).
# X is fixed at 140 mm. You input Z (mm) each time. Everything is saved to a CSV.

import cv2
import csv
import time
import numpy as np
from datetime import datetime

# ---- Constants you can tweak -------------------------------------------------
X_MM = 140.0                 # Real, physical HEIGHT of your box/marker (mm)
CAM_INDEX = 0                # Webcam index
FRAME_W, FRAME_H = 1280, 720 # Keep constant during data collection
OUT_CSV = "focal_height_measurements.csv"
ARUCO_DICT = cv2.aruco.DICT_6X6_250  # Use the dictionary you printed
# -----------------------------------------------------------------------------

def open_cam(index=0, width=1280, height=720):
    cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Try a different index or check permissions.")
    return cap

def detect_largest_marker_corners(frame_bgr):
    """Return corners (4x1x2) of the largest detected marker, or None."""
    aruco = cv2.aruco
    dictionary = aruco.getPredefinedDictionary(ARUCO_DICT)
    params = aruco.DetectorParameters_create()

    # Make detection a bit more forgiving for small/distant markers
    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeMax = 53
    params.adaptiveThreshWinSizeStep = 4
    params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
    params.minMarkerPerimeterRate = 0.02
    params.polygonalApproxAccuracyRate = 0.03
    params.minCornerDistanceRate = 0.02
    params.minDistanceToBorder = 2

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    corners_list, ids, _ = aruco.detectMarkers(gray, dictionary, parameters=params)
    if ids is None or len(corners_list) == 0:
        return None

    # Choose the marker with the largest area
    areas = [cv2.contourArea(c.reshape(-1,1,2).astype(np.float32)) for c in corners_list]
    idx = int(np.argmax(areas))
    return corners_list[idx]  # shape (1, 4, 2)

def height_pixels_from_corners(corners):
    """
    Compute HEIGHT in pixels using ONLY vertical edges.
    OpenCV ArUco corner order: [top-left, top-right, bottom-right, bottom-left].
    Height = average of left and right vertical edge lengths.
    """
    c = corners.reshape(-1, 2)
    # vertical edges: TL->BL (0->3), TR->BR (1->2)
    d_left  = np.linalg.norm(c[0] - c[3])
    d_right = np.linalg.norm(c[1] - c[2])
    return float((d_left + d_right) * 0.5)

def annotate_preview(frame, corners, height_px, f_px):
    if corners is not None:
        cv2.aruco.drawDetectedMarkers(frame, [corners])
        center = corners.reshape(-1,2).mean(axis=0)
        cv2.circle(frame, (int(center[0]), int(center[1])), 6, (0,255,0), -1)
    txt = f"h≈{height_px:.1f}px  f≈{f_px:.1f}px" if f_px is not None else "No marker"
    cv2.putText(frame, txt, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 2, cv2.LINE_AA)
    return frame

def main():
    print("=== Focal length estimation (HEIGHT only) ===")
    print(f"Assuming real marker height X = {X_MM:.1f} mm.")
    print("Instructions:")
    print(" - Place the marker flat and facing the camera (perpendicular).")
    print(" - Keep the same camera resolution for all samples.")
    print(" - For each sample, type the distance Z (mm) and press Enter; blank = finish.\n")

    cap = open_cam(CAM_INDEX, FRAME_W, FRAME_H)

    # Prepare CSV with header
    rows = [("timestamp", "sample", "Z_mm", "height_px", "f_px")]
    sample_idx = 1

    try:
        while True:
            Z_s = input("Distance Z in mm (blank to finish): ").strip()
            if Z_s == "":
                break
            try:
                Z_mm = float(Z_s)
            except ValueError:
                print("  Not a number. Try again.")
                continue

            # Grab a frame
            ok, frame = cap.read()
            if not ok:
                print("  Failed to read frame. Try again.")
                continue

            # Detect and measure height in pixels
            corners = detect_largest_marker_corners(frame)
            if corners is None:
                print("  No marker detected. Adjust lighting/position and retry.")
                # show the raw frame briefly to help aiming
                cv2.imshow("No marker detected - adjust and press any key", frame)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                continue

            height_px = height_pixels_from_corners(corners)
            f_px = (height_px * Z_mm) / X_MM

            # Visual confirmation
            preview = annotate_preview(frame.copy(), corners, height_px, f_px)
            cv2.imshow("Measurement preview", preview)
            cv2.waitKey(600)
            cv2.destroyAllWindows()

            ts = datetime.now().isoformat(timespec="seconds")
            rows.append((ts, sample_idx, Z_mm, height_px, f_px))
            print(f"  saved: sample {sample_idx}  Z={Z_mm:.0f} mm  h={height_px:.1f} px  f={f_px:.1f} px\n")
            sample_idx += 1

    finally:
        cap.release()
        cv2.destroyAllWindows()

    # Compute stats
    f_vals = np.array([r[4] for r in rows[1:]], dtype=float) if len(rows) > 1 else np.array([])
    if f_vals.size:
        f_mean = float(f_vals.mean())
        f_std  = float(f_vals.std(ddof=1)) if f_vals.size > 1 else 0.0
        rows.append(("SUMMARY", "", "", "MEAN_f_px", f_mean))
        rows.append(("SUMMARY", "", "", "STD_f_px",  f_std))

        print("\n=== Results ===")
        print(f"Samples: {len(f_vals)}")
        print(f"Mean f = {f_mean:.1f} px")
        print(f"Std  f = {f_std:.1f} px")
    else:
        print("\nNo samples recorded.")

    # Write CSV
    with open(OUT_CSV, "w", newline="") as fh:
        csv.writer(fh).writerows(rows)
    print(f"\nWrote results to {OUT_CSV}")
    print("Done.")

if __name__ == "__main__":
    main()
