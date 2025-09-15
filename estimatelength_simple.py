#!/usr/bin/env python3
"""
capture_and_measure.py
----------------------
- Headless (no GUI) capture from the Pi camera (picamera2 preferred, else GStreamer/OpenCV).
- Waits for you to press ENTER before each shot ("only take pictures when ready").
- Detects an ArUco marker (DICT_6X6_250) and measures *vertical* pixel height x_px.
- Saves each image + a CSV row with: timestamp, Z_mm (if provided), x_px, f_px (if Z & X given), file path.
- Optionally scp-pushes each saved image to your laptop (Mac) after capture.

Examples
--------
# Just capture & measure (marker is 140 mm tall), 10 distances:
python3 capture_and_measure.py --marker-mm 140 --dist-mm 300 400 500 600 700 800 900 1000 1100 1200 \
  --out /home/pi/Robotlab/Robotlab/captures

# Same, but also push to your Mac Desktop (enable Remote Login on macOS first!)
python3 capture_and_measure.py --marker-mm 140 --dist-mm 300 500 800 1200 \
  --out /home/pi/Robotlab/Robotlab/captures \
  --push "noah@<YOUR-MAC-IP>:/Users/noah/Desktop/ArloCaptures"
"""

import argparse, os, sys, time, csv, subprocess
from datetime import datetime
import numpy as np

def make_camera(width=1640, height=1232, fps=30):
    """Return (read_fn, release_fn) abstracting the camera."""
    # Prefer picamera2 on the Pi
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
        # Fallback: GStreamer pipeline similar to course examples
        import cv2
        def gstreamer_pipeline(capture_width=1024, capture_height=720, framerate=30):
            return (
                "libcamerasrc ! "
                "videobox autocrop=true ! "
                f"video/x-raw, width=(int){capture_width}, height=(int){capture_height}, framerate=(fraction){framerate}/1 ! "
                "videoconvert ! appsink"
            )
        cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
        if not cap.isOpened():
            raise RuntimeError(f"Camera init failed (picamera2 error: {e})")

        def read_fn():
            ok, frame = cap.read()
            return ok, frame

        def release_fn():
            try: cap.release()
            except: pass

        return read_fn, release_fn

def detect_aruco_bgr(frame_bgr, restrict_id=None):
    """Detect ArUco (DICT_6X6_250); return {'x_px','id','corners'} or None."""
    import cv2
    aruco = cv2.aruco
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
    parameters = aruco.DetectorParameters_create()
    corners_list, ids, _ = aruco.detectMarkers(frame_bgr, dictionary, parameters=parameters)
    if ids is None or len(corners_list) == 0:
        return None
    best = None
    for c, idval in zip(corners_list, ids.flatten()):
        if restrict_id is not None and idval != restrict_id:
            continue
        pts = c.reshape(-1,2)  # TL, TR, BR, BL
        TL, TR, BR, BL = pts
        v1 = np.linalg.norm(TL - BL)
        v2 = np.linalg.norm(TR - BR)
        x_px = 0.5*(v1+v2)     # vertical pixel height
        if (best is None) or (x_px > best[0]):
            best = (x_px, int(idval), pts)
    if best is None:
        return None
    return {"x_px": float(best[0]), "id": best[1], "corners": best[2]}

def annotate_and_save(frame_bgr, det, text, out_dir, basename):
    import cv2
    img = frame_bgr.copy()
    if det is not None:
        cv2.polylines(img, [det["corners"].astype(int)], True, (0,255,0), 2)
    cv2.putText(img, text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    path = os.path.join(out_dir, basename)
    ok = cv2.imwrite(path, img)
    if not ok:
        print(f"[WARN] Could not write image: {path}")
    return os.path.abspath(path)

def ensure_csv(out_dir):
    path = os.path.join(out_dir, "results.csv")
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["timestamp","Z_mm","x_px","f_px","marker_id","filename"])
    return path

def scp_push(local_path, remote_dest):
    """Push a file to remote via scp. remote_dest like 'noah@<ip>:/Users/noah/Desktop/ArloCaptures'."""
    try:
        os.makedirs(os.path.expanduser("~/.ssh"), exist_ok=True)  # ensure ssh dir exists
        # -q quiet, -o options avoid interactivity on first use can be disabled if you want strict
        cmd = ["scp","-q",local_path, remote_dest]
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"[WARN] scp failed for {local_path} -> {remote_dest}: {e}")
        return False

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--marker-mm", type=float, default=140.0, help="Physical marker height X in mm (you said 14 cm).")
    p.add_argument("--dist-mm", type=float, nargs="*", default=None, help="Optional list of distances Z in mm (one capture per value).")
    p.add_argument("--count", type=int, default=10, help="If no --dist-mm, take this many shots.")
    p.add_argument("--id", type=int, default=None, help="Optional: only accept a specific ArUco ID.")
    p.add_argument("--out", type=str, required=True, help="Output directory on the Pi.")
    p.add_argument("--push", type=str, default=None, help="Optional scp destination (e.g., noah@<MAC-IP>:/Users/noah/Desktop/ArloCaptures)")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    csv_path = ensure_csv(args.out)

    read_fn, release_fn = make_camera()

    print("\nHeadless capture. Place the box/marker, then press ENTER to shoot.")
    if args.dist_mm:
        plan = [float(z) for z in args.dist_mm]
        print(f"Planned distances: {plan}")
    else:
        plan = [None]*args.count
        print(f"Planned shots (no Z known): {args.count}")
    print("Ctrl+C to abort.\n")

    idx = 0
    rows = []
    try:
        for Z in plan:
            prompt = f"[Z={Z:.1f} mm]" if Z is not None else "[no Z]"
            input(f"{prompt}  Press ENTER to capture...")

            ok, frame = read_fn()
            if not ok:
                print("[ERR] Camera read failed; retrying…")
                time.sleep(0.2)
                ok, frame = read_fn()
                if not ok:
                    print("[ERR] Still failing; skipping.")
                    continue

            det = detect_aruco_bgr(frame, restrict_id=args.id)
            if det is None:
                print("No marker detected — shot saved without measurement.")
                x_px = None; f_px = None; mid = None
                overlay = f"{prompt} (NO MARKER)"
            else:
                x_px = det["x_px"]
                mid  = det["id"]
                f_px = (x_px * Z / args.marker_mm) if Z is not None else None
                overlay = f"{prompt} x={x_px:.1f}px" + (f", f={f_px:.1f}px" if f_px is not None else "")

            idx += 1
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            base = f"capture_{idx:02d}" + (f"_Z{int(Z)}" if Z is not None else "") + ".png"
            saved = annotate_and_save(frame, det, overlay, args.out, base)

            # Append CSV
            with open(csv_path, "a", newline="") as f:
                w = csv.writer(f)
                w.writerow([ts, f"{Z:.1f}" if Z is not None else "", f"{x_px:.2f}" if x_px is not None else "",
                            f"{f_px:.2f}" if f_px is not None else "", mid if mid is not None else "", os.path.basename(saved)])

            print(f"Saved: {saved}")
            if args.push:
                ok = scp_push(saved, args.push)
                if ok:
                    print(f"Pushed to: {args.push}")

            rows.append((Z, x_px, f_px, mid, saved))

    except KeyboardInterrupt:
        print("\nInterrupted; finishing up…")
    finally:
        release_fn()

    print("\nSummary:")
    print(f"{'shot':>4}  {'Z(mm)':>8}  {'x(px)':>10}  {'f(px)':>10}  {'ID':>4}  file")
    for i,(Z,x_px,f_px,mid,path) in enumerate(rows, start=1):
        ztxt = f"{Z:.1f}" if Z is not None else "-"
        xtxt = f"{x_px:.2f}" if x_px is not None else "-"
        ftxt = f"{f_px:.2f}" if f_px is not None else "-"
        idtxt = f"{mid}" if mid is not None else "-"
        print(f"{i:4d}  {ztxt:>8}  {xtxt:>10}  {ftxt:>10}  {idtxt:>4}  {os.path.basename(path)}")

    print(f"\nCSV written to: {csv_path}")
    if args.push:
        print("Note: CSV is local on the Pi; you can scp it later if you want.")

if __name__ == "__main__":
    main()
