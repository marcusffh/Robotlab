import cv2
import time
import os

def gstreamer_pipeline(capture_width=1024, capture_height=720, framerate=30):
    return (
        "libcamerasrc ! "
        "videobox autocrop=true ! "
        f"video/x-raw, width=(int){capture_width}, height=(int){capture_height}, framerate=(fraction){framerate}/1 ! "
        "videoconvert ! appsink"
    )

def capture_images(outdir="captures", num_images=10, delay=2):
    # Make output folder if it doesn’t exist
    os.makedirs(outdir, exist_ok=True)

    # Open the camera
    cam = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
    if not cam.isOpened():
        print("Error: Could not open camera.")
        return

    print(f"Capturing {num_images} images...")
    for i in range(num_images):
        time.sleep(delay)  # wait between captures (you can move the box meanwhile)
        ret, frame = cam.read()
        if not ret:
            print("Failed to capture frame")
            continue
        filename = os.path.join(outdir, f"image_{i+1}.png")
        cv2.imwrite(filename, frame)
        print(f"Saved {filename}")

    cam.release()
    print("Done.")

if __name__ == "__main__":
    capture_images(outdir="captures", num_images=10, delay=3)
