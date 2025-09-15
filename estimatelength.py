#!/usr/bin/env python3
"""
Exercise 3 - Focal length estimation with ArUco
------------------------------------------------
This script captures an ArUco marker at known distances Z (mm)
with known physical edge size X (mm), measures its pixel size x,
and computes f = x * Z / X. Results are saved to CSV, plots, and
a LaTeX table for inclusion in the report.

Controls:
  - Press 'c' to capture at current distance
  - Press 'n' to skip
  - Press 'q' to quit
"""

import cv2
import numpy as np
import argparse
import os, time, csv
from datetime import datetime
import matplotlib.pyplot as plt

# --- Camera setup ---
def make_camera(width=1640, height=1232, fps=30):
    try:
        from picamera2 import Picamera2
        cam = Picamera2()
        frame_dur = int(1.0/fps*1_000_000)
        config = cam.create_video_configuration(
            main={"size": (width,height), "format":"RGB888"},
            controls={"FrameDurationLimits": (frame_dur, frame_dur)},
            queue=False)
        cam.configure(config)
        cam.start()
        time.sleep(1.0)
        def read_fn():
            frame = cam.capture_array("main")
            return True, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return cam, read_fn, lambda: cam.stop()
    except Exception:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("No camera found!")
        def read_fn():
            return cap.read()
        return cap, read_fn, lambda: cap.release()

# --- ArUco detection ---
def detect_marker(frame, restrict_id=None):
    aruco = cv2.aruco
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
    parameters = aruco.DetectorParameters_create()
    corners, ids, _ = aruco.detectMarkers(frame, dictionary, parameters=parameters)
    if ids is None or len(corners)==0:
        return None
    best = None
    for c,idval in zip(corners, ids.flatten()):
        if restrict_id and idval!=restrict_id: continue
        pts = c.reshape(-1,2)
        TL,TR,BR,BL = pts
        v1,v2 = np.linalg.norm(TL-BL), np.linalg.norm(TR-BR)
        x_px = 0.5*(v1+v2)
        if best is None or x_px>best[0]:
            best=(x_px,idval,pts)
    if best: return {"x_px":best[0],"id":best[1],"corners":best[2]}
    return None

# --- Main ---
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--marker-mm",type=float,required=True,help="Edge size X of marker in mm")
    ap.add_argument("--dist-mm",type=float,nargs="+",required=True,help="List of distances Z in mm")
    ap.add_argument("--out",type=str,default="focal_results",help="Output folder")
    ap.add_argument("--id",type=int,default=None,help="Restrict to a specific marker id")
    args=ap.parse_args()

    os.makedirs(args.out,exist_ok=True)
    csv_path=os.path.join(args.out,"f_estimates.csv")
    fig_hist=os.path.join(args.out,"f_hist.png")
    fig_line=os.path.join(args.out,"f_vs_Z.png")
    tex_path=os.path.join(args.out,"f_table.tex")

    cam,read_fn,release_fn=make_camera()
    cv2.namedWindow("FocalEst")

    rows=[]
    for Z in args.dist_mm:
        print(f"\nDistance {Z} mm: press 'c' to capture, 'n' to skip, 'q' to quit.")
        while True:
            ok,frame=read_fn()
            if not ok: break
            info=detect_marker(frame,args.id)
            f_est=None
            if info: f_est=info["x_px"]*Z/args.marker_mm
            overlay=frame.copy()
            if info:
                cv2.polylines(overlay,[info["corners"].astype(int)],True,(0,255,0),2)
                cv2.putText(overlay,f"f={f_est:.1f}px",(20,40),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
            cv2.imshow("FocalEst",overlay)
            key=cv2.waitKey(1)&0xFF
            if key==ord('c') and info:
                ts=datetime.now().strftime("%Y%m%d_%H%M%S")
                row={"Z":Z,"x_px":info["x_px"],"f":f_est,"id":info["id"],"ts":ts}
                rows.append(row)
                print(f"Captured: Z={Z}, x={info['x_px']:.1f}px, f={f_est:.1f}px")
                break
            if key==ord('n') or key==ord('q'):
                break
        if key==ord('q'): break

    release_fn()
    cv2.destroyAllWindows()

    if not rows: return
    # Save CSV
    with open(csv_path,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    f_vals=[r["f"] for r in rows]; mean=np.mean(f_vals); std=np.std(f_vals,ddof=1)
    print(f"Mean f={mean:.2f}px, std={std:.2f}px")

    # Figures
    plt.hist(f_vals,bins=min(10,len(f_vals)))
    plt.axvline(mean,linestyle="--",color="red")
    plt.title("Distribution of f estimates"); plt.xlabel("f (px)"); plt.ylabel("count")
    plt.savefig(fig_hist); plt.close()

    Zs=[r["Z"] for r in rows]
    plt.plot(Zs,f_vals,"o-"); plt.axhline(mean,linestyle="--",color="red")
    plt.title("f vs distance"); plt.xlabel("Z (mm)"); plt.ylabel("f (px)")
    plt.savefig(fig_line); plt.close()

    # LaTeX table
    with open(tex_path,"w") as f:
        f.write("\\begin{table}[h]\n\\centering\n")
        f.write("\\begin{tabular}{r r r r}\n\\hline\n")
        f.write("Z (mm) & x (px) & f (px) & ID \\\\\\hline\n")
        for r in rows:
            f.write(f"{r['Z']} & {r['x_px']:.1f} & {r['f']:.1f} & {r['id']} \\\\\n")
        f.write(f"\\hline\n\\multicolumn{{4}}{{r}}{{Mean f={mean:.1f}px, Std={std:.1f}px}} \\\\\n")
        f.write("\\hline\n\\end{tabular}\n")
        f.write("\\caption{Focal length estimates across distances}\n")
        f.write("\\label{tab:focal}\n\\end{table}\n")
    print(f"Saved CSV, figures, and LaTeX table in {args.out}")

if __name__=="__main__":
    main()
