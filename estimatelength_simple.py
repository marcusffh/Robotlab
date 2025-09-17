#!/usr/bin/env python3
"""
focal_length_live.py — Headless focal length estimation (picamera + OpenCV)

- Uses picamera for image capture (no GUI; works over SSH).
- For each provided distance Z (mm), waits for ENTER, grabs a frame,
  detects an ArUco (DICT_6X6_250), measures the marker's vertical pixel
  height x, and computes f = x * Z / X where X is the marker height in mm.
- Prints a final table + mean and standard deviation of f.

Usage (your distances 1.0..5.5 m → in mm):
  python3 focal_length_live.py \
    --marker-mm 140 \
    --dist-mm 1000 1500 2000 2500 3000 3500 4000 4500 5000 5500
"""

import argparse, time, sys
import numpy as np
import cv2

# ---------- Camera (picamera) ----------
def make_camera(width=1640, height=1232, fps=30):
    """
    Return (read_fn, release_fn) using the legacy 'picamera' module.
    Frames are returned as BGR numpy arrays suitable for OpenCV.
    """
    try:
        from picamera import PiCamera
        from picamera.array import PiRGBArray
    except ImportError as e:
        raise RuntimeError(
            "picamera is required for capture in this refactor. "
            "Install it (sudo apt install python3-picamera) and enable the legacy camera stack."
        ) from e

    cam = PiCamera()
    cam.resolution = (width, height)
    cam.framerate = fps
    # Let the sensor warm up for more stable exposure/white balance
    time.sleep(1.0)

    raw = PiRGBArray(cam, size=(width, height))

    def read_fn():
        # Capture directly in BGR so no color conversion is needed
        cam.capture(raw, format="bgr", use_video_port=True)
        frame = raw.array
        raw.truncate(0)   # reset the stream for the next capture
        return True, frame

    def release_fn():
        try:
            raw.close()
        finally:
            cam.close()

    return read_fn, release_fn

# ---------- ArUco detection (OpenCV) ----------
def detect_aruco_vertical_px(frame_bgr, restrict_id=None):
    """
    Detect an ArUco marker (DICT_6X6_250) in BGR frame.
    Returns (x_px, marker_id) where x_px is the vertical pixel height
    (mean of left and right edges). If none found, returns (None, None).
    """
    aruco = cv2.aruco
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
    parameters = aruco.DetectorParameters_create()

    corners_list, ids, _ = aruco.detectMarkers(frame_bgr, dictionary, parameters=parameters)
    if ids is None or len(corners_list) == 0:
        return None, None

    best = None
    for corners, mid in zip(corners_list, ids.flatten()):
        if restrict_id is not None and int(mid) != restrict_id:
            continue
        pts = corners.reshape(-1, 2)  # TL, TR, BR, BL
        TL, TR, BR, BL = pts
        v1 = np.linalg.norm(TL - BL)
        v2 = np.linalg.norm(TR - BR)
        x_px = 0.5 * (v1 + v2)
        if (best is None) or (x_px > best[0]):
            best = (float(x_px), int(mid))

    if best is None:
        return None, None
    return best

# ---------- Main CLI ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--marker-mm", type=float, required=True,
                    help="Physical marker height X in mm (e.g., 140 for 14 cm).")
    ap.add_argument("--dist-mm", type=float, nargs="+", required=True,
                    help="Distances Z in mm, e.g., 1000 1500 ... 5500.")
    ap.add_argument("--id", type=int, default=None,
                    help="Optional: restrict to a specific ArUco ID.")
    args = ap.parse_args()

    read_fn, release_fn = make_camera()

    rows = []  # (Z_mm, x_px, f_px, marker_id)
    print("\nFocal length estimation — headless")
    print("For each distance below, place the marker and press ENTER to capture.\n")

    try:
        for Z in args.dist_mm:
            input(f"[Z = {Z:.1f} mm]  Press ENTER to capture...")
            ok, frame = read_fn()
            if not ok:
                print("  [ERR] Camera read failed; skipping this distance.")
                continue

            x_px, mid = detect_aruco_vertical_px(frame, restrict_id=args.id)
            while x_px is None:
                print("  No marker detected — adjust pose/lighting and press ENTER to retry...")
                input()
                ok, frame = read_fn()
                if not ok:
                    print("  [ERR] Camera read failed; retrying…"); continue
                x_px, mid = detect_aruco_vertical_px(frame, restrict_id=args.id)

            f_px = (x_px * Z) / args.marker_mm
            rows.append((Z, x_px, f_px, mid))
            print(f"  OK: x={x_px:.2f} px  ->  f={f_px:.2f} px  (ID={mid})")

    except KeyboardInterrupt:
        print("\nInterrupted; finishing up…")
    finally:
        release_fn()

    if not rows:
        print("\nNo measurements collected.")
        sys.exit(0)

    # Mean / std of f
    f_vals = np.array([r[2] for r in rows], dtype=float)
    f_mean = float(np.mean(f_vals))
    f_std = float(np.std(f_vals, ddof=1)) if len(f_vals) > 1 else 0.0

    # Final table
    print("\nResults:")
    print(f"{'Z (mm)':>8}  {'x (px)':>10}  {'f (px)':>10}  {'ID':>4}")
    for Z, x_px, f_px, mid in rows:
        print(f"{Z:8.1f}  {x_px:10.2f}  {f_px:10.2f}  {mid:4d}")

    print(f"\nMean f = {f_mean:.2f} px")
    print(f"Std  f = {f_std:.2f} px")
    print("\nUse the mean f for the next parts; report the std as your precision.")

if __name__ == "__main__":
    main()
