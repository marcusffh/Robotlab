# part1_estimate_f_webcam.py
import cv2, csv, numpy as np, pathlib
from aruco_utils import detect_markers, largest_marker_index, side_len_pixels

def open_cam(index=0, width=1280, height=720):
    cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")
    return cap

def main():
    print("=== Focal-length estimation ===")
    X_mm = float(input("Enter marker side length X in mm: ").strip())
    out = pathlib.Path("f_measurements_webcam.csv")
    rows = [("Z_mm", "x_px", "f_px")]

    cap = open_cam()
    print("\nInstructions:")
    print("1) Place the marker flat, centered in view at the distance Z you will input.")
    print("2) For each measurement, type the distance Z in mm and press Enter.")
    print("   Leave blank and press Enter when you are DONE.\n")

    while True:
        Z_s = input("Distance Z in mm (blank to finish): ").strip()
        if Z_s == "":
            break
        try:
            Z_mm = float(Z_s)
        except ValueError:
            print("  Not a number")
            continue

        # Grab a frame
        ok, frame = cap.read()
        if not ok:
            print("  Failed to read frame")
            continue

        corners, ids = detect_markers(frame)
        if ids is None:
            print("  No ArUco marker found")
            # show the live frame to help user adjust
            cv2.imshow("Adjust marker (press any key to continue)", frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            continue

        idx = largest_marker_index(corners)
        x_px = side_len_pixels(corners[idx])
        f_px = x_px * Z_mm / X_mm
        rows.append((Z_mm, x_px, f_px))

        # Visual feedback
        c = corners[idx].reshape(-1,2).mean(axis=0)
        cv2.circle(frame, (int(c[0]), int(c[1])), 6, (0,255,0), -1)
        cv2.putText(frame, f"x≈{x_px:.1f}px  f≈{f_px:.1f}px", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 2, cv2.LINE_AA)
        cv2.imshow("Measurement", frame)
        cv2.waitKey(600)
        cv2.destroyAllWindows()

        f_vals = np.array([r[2] for r in rows[1:]], dtype=float)
        print(f"  saved: Z={Z_mm:.0f} mm, x={x_px:.1f} px, f={f_px:.1f} px")
        print(f"  current mean f={f_vals.mean():.1f} px, std={f_vals.std(ddof=1) if len(f_vals)>1 else 0:.1f} px, n={len(f_vals)}\n")

    cap.release()

    with out.open("w", newline="") as fh:
        csv.writer(fh).writerows(rows)

    if len(rows) > 1:
        f_vals = np.array([r[2] for r in rows[1:]], dtype=float)
        print(f"\nDone. Wrote {len(rows)-1} samples → {out}")
        print(f"Recommended f (mean): {f_vals.mean():.1f} px  (std: {f_vals.std(ddof=1) if len(f_vals)>1 else 0:.1f})")
    else:
        print("\nNo samples recorded.")

if __name__ == "__main__":
    main()
