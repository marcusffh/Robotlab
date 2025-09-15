#!/usr/bin/env python3
"""
capture_series_cli.py  —  Headless image capture for focal-length exercise
---------------------------------------------------------------------------
Takes one photo per provided distance Z (mm). No GUI. Perfect for SSH.

Usage:
  python3 capture_series_cli.py --out /home/pi/Robotlab/Robotlab/captures \
      --dist-mm 300 400 500 600 700 800 900 1000 1100 1200

Then, for each Z, place your box/marker at that distance and press ENTER.
The script saves:
  - PNG images: capture_Z<Z>_idx<k>.png
  - manifest.csv with columns: Z_mm, filename, timestamp
"""

import argparse, os, time, csv
from datetime import datetime

def make_camera(width=1640, height=1232, fps=30):
    """Return (read_fn, release_fn) abstracting the camera.
       read_fn() -> (ok, frame_bgr) ; release_fn() -> None
    """
    # Prefer picamera2 for best Pi results
    try:
        from picamera2 import Picamera2
        import cv2
        cam = Picamera2()
        frame_dur = int(1.0/fps*1_000_000)
        cfg = cam.create_video_configuration(
            main={"size": (width, height), "format": "RGB888"},
            controls={"FrameDurationLimits": (frame_dur, frame_dur)},
            queue=False
        )
        cam.configure(cfg)
        cam.start()
        time.sleep(1.0)
        def read_fn():
            rgb = cam.capture_array("main")  # RGB
            return True, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        def release_fn():
            try: cam.stop()
            except: pass
        return read_fn, release_fn
    except Exception as e:
        # Fallback to OpenCV VideoCapture (USB cam or libcamera appsink)
        import cv2
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
        if not cap.isOpened():
            raise RuntimeError(f"No camera available (picamera2 failed: {e})")
        def read_fn():
            ok, frame = cap.read()
            return ok, frame
        def release_fn():
            try: cap.release()
            except: pass
        return read_fn, release_fn

def ensure_manifest(out_dir):
    path = os.path.join(out_dir, "manifest.csv")
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["timestamp", "Z_mm", "filename"])
    return path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, required=True, help="Output folder (absolute path recommended)")
    ap.add_argument("--dist-mm", type=float, nargs="+", required=True, help="Distances Z in mm (one photo per Z)")
    ap.add_argument("--width", type=int, default=1640, help="Capture width (px)")
    ap.add_argument("--height", type=int, default=1232, help="Capture height (px)")
    ap.add_argument("--fps", type=int, default=30, help="Capture FPS (used for exposure timing)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    manifest_path = ensure_manifest(args.out)

    read_fn, release_fn = make_camera(args.width, args.height, args.fps)

    print("Headless capture. For each Z, press ENTER to take one photo.")
    print("Tip: keep lighting steady; hold the box perpendicular to the camera.")
    print()

    idx = 0
    try:
        for Z in args.dist_mm:
            input(f"[Z={Z:.1f} mm] Place the box at this distance, then press ENTER to capture...")
            ok, frame = read_fn()
            if not ok:
                print("[ERR] Camera read failed; try this Z again.")
                # simple retry once
                time.sleep(0.2)
                ok, frame = read_fn()
                if not ok:
                    print("[ERR] Still failing; skipping this Z.")
                    continue

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            idx += 1
            fname = f"capture_Z{int(Z)}_idx{idx}.png"
            fpath = os.path.join(args.out, fname)

            # Save PNG
            import cv2
            ok = cv2.imwrite(fpath, frame)
            if not ok:
                print(f"[WARN] Could not write image: {fpath}")
            else:
                # Append to manifest
                with open(manifest_path, "a", newline="") as f:
                    w = csv.writer(f)
                    w.writerow([ts, f"{Z:.1f}", fname])
                print(f"Saved: {fpath}")

    except KeyboardInterrupt:
        print("\nInterrupted. Finishing up…")
    finally:
        release_fn()

    print("\nDone. Files are in:", os.path.abspath(args.out))
    print("Manifest:", os.path.abspath(manifest_path))
    print("Next: measure pixel height x in each image (e.g., in OpenCV or ImageJ) and compute f = x*Z/X.")

if __name__ == "__main__":
    main()
