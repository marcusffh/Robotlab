#!/usr/bin/env python3
import argparse, time, sys
import numpy as np
import cv2

def make_camera(width=1920, height=1080, fps=30):
    """Use picamera2 for capture; return (read_fn, release_fn)."""
    from picamera2 import Picamera2
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
        # returns BGR for OpenCV
        rgb = cam.capture_array("main")
        return True, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    def release_fn():
        try: cam.stop()
        except: pass
    return read_fn, release_fn

def detect_aruco_vertical_px(frame_bgr, restrict_id=None):
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
        x_px = 0.5 * (v1 + v2)
        if (best is None) or (x_px > best[0]):
            best = (float(x_px), int(mid))
    return best if best is not None else (None, None)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--marker-mm", type=float, required=True)
    ap.add_argument("--dist-mm", type=float, nargs="+", required=True)
    ap.add_argument("--id", type=int, default=None)
    args = ap.parse_args()

    read_fn, release_fn = make_camera()
    rows = []
    print("\nFocal length estimation — headless")
    print("For each distance below, place the marker and press ENTER to capture.\n")
    try:
        for Z in args.dist_mm:
            input(f"[Z = {Z:.1f} mm]  Press ENTER to capture...")
            ok, frame = read_fn()
            if not ok:
                print("  [ERR] Camera read failed; skipping.")
                continue
            x_px, mid = detect_aruco_vertical_px(frame, restrict_id=args.id)
            while x_px is None:
                print("  No marker detected — adjust pose/lighting and press ENTER to retry...")
                input()
                ok, frame = read_fn()
                if not ok: 
                    print("  [ERR] Camera read failed; retrying…"); 
                    continue
                x_px, mid = detect_aruco_vertical_px(frame, restrict_id=args.id)
            f_px = (x_px * Z) / args.marker_mm
            rows.append((Z, x_px, f_px, mid))
            print(f"  OK: x={x_px:.2f} px  ->  f={f_px:.2f} px  (ID={mid})")
    except KeyboardInterrupt:
        print("\nInterrupted; finishing up…")
    finally:
        release_fn()

    if not rows:
        print("\nNo measurements collected."); sys.exit(0)

    f_vals = np.array([r[2] for r in rows], dtype=float)
    f_mean = float(np.mean(f_vals))
    f_std  = float(np.std(f_vals, ddof=1)) if len(f_vals) > 1 else 0.0

    print("\nResults:")
    print(f"{'Z (mm)':>8}  {'x (px)':>10}  {'f (px)':>10}  {'ID':>4}")
    for Z, x_px, f_px, mid in rows:
        print(f"{Z:8.1f}  {x_px:10.2f}  {f_px:10.2f}  {mid:4d}")
    print(f"\nMean f = {f_mean:.2f} px")
    print(f"Std  f = {f_std:.2f} px")
    print("\nUse the mean f for the next parts; report the std as your precision.")

if __name__ == "__main__":
    main()
