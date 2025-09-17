#!/usr/bin/env python3
"""
focal_length_live.py — Headless focal length estimation (Picamera2 only)

- Works over SSH (no GUI).
- For each provided distance Z (mm), waits for ENTER, grabs a frame,
  detects an ArUco (DICT_6X6_250), measures the marker's vertical pixel
  height x, and computes f = x * Z / X where X is the marker height in mm.
- Prints a final table + mean and standard deviation of f.
"""

import argparse, time, sys
import numpy as np
import cv2

# ---------- Camera (Picamera2 ONLY) ----------
def make_camera(width=1640, height=1232, fps=30):
    """Return (read_fn, release_fn) using Picamera2 only. Raise if unavailable."""
    try:
        from picamera2 import Picamera2
    except ImportError as e:
        raise RuntimeError("Picamera2 is not installed / importable.") from e

    cam = Picamera2()
    frame_dur_us = int(1.0 / fps * 1_000_000)
    cfg = cam.create_video_configuration(
        main={"size": (width, height), "format": "RGB888"},
        controls={"FrameDurationLimits": (frame_dur_us, frame_dur_us)},
        queue=False
    )
    cam.configure(cfg)
    cam.start()
    time.sleep(1.0)  # warm-up for exposure/gain

    def read_fn():
        rgb = cam.capture_array("main")
        if rgb is None or rgb.size == 0:
            return False, None
        # Convert to BGR for OpenCV processing
        return True, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    def release_fn():
        try:
            cam.stop()
        except Exception:
            pass

    return read_fn, release_fn

# ---------- ArUco detection (OpenCV) ----------
def detect_aruco_vertical_px(frame_bgr, restrict_id=None):
    """
    Detect ArUco (DICT_6X6_250). Return (x_px, marker_id) or (None, None).
    x_px is the average vertical pixel height of the marker (edge TL-BL and TR-BR).
    """
    aruco = cv2.aruco
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
    params = aruco.DetectorParameters_create()

    # Grayscale improves robustness on small/low-contrast tags
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    corners_list, ids, _ = aruco.detectMarkers(gray, dictionary, parameters=params)
    if ids is None or len(corners_list) == 0:
        return None, None

    best = None
    for c, mid in zip(corners_list, ids.flatten()):
        if restrict_id is not None and int(mid) != restrict_id:
            continue
        pts = c.reshape(-1, 2)  # TL, TR, BR, BL
        TL, TR, BR, BL = pts
        v1 = np.linalg.norm(TL - BL)
        v2 = np.linalg.norm(TR - BR)
        x_px = 0.5 * (v1 + v2)
        if (best is None) or (x_px > best[0]):
            best = (x_px, int(mid))

    if best is None:
        return None, None
    return float(best[0]), best[1]

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--marker-mm", type=float, required=True,
                    help="Physical marker height X in mm (e.g., 140 for 14 cm).")
    ap.add_argument("--dist-mm", type=float, nargs="+", required=True,
                    help="Distances Z in mm, e.g., 1000 1500 ... 5500.")
    ap.add_argument("--id", type=int, default=None,
                    help="Optional: restrict to a specific ArUco ID.")
    ap.add_argument("--width", type=int, default=1640)
    ap.add_argument("--height", type=int, default=1232)
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()

    read_fn, release_fn = make_camera(args.width, args.height, args.fps)

    rows = []  # (Z_mm, x_px, f_px, marker_id)
    print("\nFocal length estimation — Picamera2 + OpenCV (no fallback)")
    print("For each distance below, place the marker and press ENTER to capture.\n")

    try:
        for Z in args.dist_mm:
            input(f"[Z = {Z:.1f} mm]  Press ENTER to capture...")
            ok, frame = read_fn()
            if not ok:
                raise RuntimeError("Camera read failed (Picamera2).")

            x_px, mid = detect_aruco_vertical_px(frame, restrict_id=args.id)
            while x_px is None:
                print("  No marker detected — adjust pose/lighting and press ENTER to retry...")
                input()
                ok, frame = read_fn()
                if not ok:
                    raise RuntimeError("Camera read failed (Picamera2).")
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

    f_vals = np.array([r[2] for r in rows], dtype=float)
    f_mean = float(np.mean(f_vals))
    f_std  = float(np.std(f_vals, ddof=1)) if len(f_vals) > 1 else 0.0

    print("\nResults:")
    print(f"{'Z (mm)':>8}  {'x (px)':>10}  {'f (px)':>10}  {'ID':>4}")
    for Z, x_px, f_px, mid in rows:
        print(f"{Z:8.1f}  {x_px:10.2f}  {f_px:10.2f}  {mid:4d}")

    print(f"\nMean f = {f_mean:.2f} px")
    print(f"Std  f = {f_std:.2f} px")
    print("\nUse the mean f for later exercises; report the std as precision.")

if __name__ == "__main__":
    main()
