#!/usr/bin/env python3
"""
focal_length_live.py — Headless focal length estimation (no image saving)

- Works over SSH (no GUI).
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

def make_camera(width=1640, height=1232, fps=30):
    """Return (read_fn, release_fn) to get BGR frames, preferring picamera2."""
    try:
        from picamera2 import Picamera2
        import cv2
        cam = Picamera2()
        frame_dur = int(1.0/fps * 1_000_000)
        cfg = cam.create_video_configuration(
            main={"size": (width, height), "format": "RGB888"},
            controls={"FrameDurationLimits": (frame_dur, frame_dur)},
            queue=False
        )
        cam.configure(cfg)
        cam.start()
        time.sleep(1.0)
        def read_fn():
            rgb = cam.capture_array("main")
            return True, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        def release_fn():
            try: cam.stop()
            except: pass
        return read_fn, release_fn
    except Exception as e:
        import cv2
        # Fallback to GStreamer pipeline used in your examples
        def gstreamer_pipeline(capture_width=1024, capture_height=720, framerate=30):
            return (
                "libcamerasrc ! "
                "videobox autocrop=true ! "
                f"video/x-raw, width=(int){capture_width}, height=(int){capture_height}, framerate=(fraction){framerate}/1 ! "
                "videoconvert ! appsink"
            )
        cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
        if not cap.isOpened():
            raise RuntimeError(f"Camera init failed (picamera2 error: {e}) and GStreamer fallback also failed")
        def read_fn():
            ok, frame = cap.read()
            return ok, frame
        def release_fn():
            try: cap.release()
            except: pass
        return read_fn, release_fn

def detect_aruco_vertical_px(frame_bgr, restrict_id=None):
    """Detect ArUco (DICT_6X6_250). Return (x_px, marker_id) or (None, None)."""
    import cv2
    aruco = cv2.aruco
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
    parameters = aruco.DetectorParameters_create()
    corners_list, ids, _ = aruco.detectMarkers(frame_bgr, dictionary, parameters=parameters)
    if ids is None or len(corners_list) == 0:
        return None, None
    best = None
    for c, mid in zip(corners_list, ids.flatten()):
        if restrict_id is not None and mid != restrict_id:
            continue
        pts = c.reshape(-1, 2)  # TL, TR, BR, BL
        TL, TR, BR, BL = pts
        v1 = np.linalg.norm(TL - BL)
        v2 = np.linalg.norm(TR - BR)
        x_px = 0.5 * (v1 + v2)   # vertical pixel height (mean of both sides)
        if (best is None) or (x_px > best[0]):
            best = (x_px, int(mid))
    if best is None:
        return None, None
    return float(best[0]), best[1]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--marker-mm", type=float, required=True, help="Physical marker height X in mm (e.g., 140 for 14 cm).")
    ap.add_argument("--dist-mm", type=float, nargs="+", required=True, help="Distances Z in mm, e.g., 1000 1500 ... 5500.")
    ap.add_argument("--id", type=int, default=None, help="Optional: restrict to a specific ArUco ID.")
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
                print("  [ERR] Camera read failed, retrying…")
                time.sleep(0.2)
                ok, frame = read_fn()
                if not ok:
                    print("  [ERR] Still failing; skipping this distance.")
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

    # Compute mean / std of f
    f_vals = np.array([r[2] for r in rows], dtype=float)
    f_mean = float(np.mean(f_vals))
    f_std = float(np.std(f_vals, ddof=1)) if len(f_vals) > 1 else 0.0

    # Print final table
    print("\nResults:")
    print(f"{'Z (mm)':>8}  {'x (px)':>10}  {'f (px)':>10}  {'ID':>4}")
    for Z, x_px, f_px, mid in rows:
        print(f"{Z:8.1f}  {x_px:10.2f}  {f_px:10.2f}  {mid:4d}")

    print(f"\nMean f = {f_mean:.2f} px")
    print(f"Std  f = {f_std:.2f} px")
    print("\nUse the mean f for the next parts; report the std as your precision.")
    # Done.
if __name__ == "__main__":
    main()
