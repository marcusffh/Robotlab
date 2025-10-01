# camera.py — unified camera backend with Picamera2 / picamera / OpenCV fallback
# Provides:
#   class Camera:
#       get_next_frame() -> BGR np.ndarray
#       detect_aruco_objects(img) -> (ids, dists_cm, angles_rad)
#       draw_aruco_objects(img) -> overlay on BGR image
#       terminateCaptureThread() -> stop worker if used
#
# Notes:
# - Requires opencv-contrib (cv2.aruco). Install one of:
#     pip install opencv-contrib-python==4.7.0.72
#     # or headless:
#     pip install opencv-contrib-python-headless==4.7.0.72

import cv2
import numpy as np
import time
import sys
import threading

# Optional ring buffer helper used by your existing code
try:
    import framebuffer
except Exception:
    framebuffer = None  # fallback if not available

# Backend discovery flags
gstreamerCameraFound = False
piCameraFound = False
piCamera2Found = False

# ---- Try Picamera2 (modern libcamera stack) ----
try:
    from picamera2 import Picamera2
    piCamera2Found = True
    print("Camera.py: picamera2 module available")
except Exception as e:
    print("Camera.py: picamera2 module not available:", e)
    piCamera2Found = False

# ---- Try legacy picamera (MMAL) ----
if not piCamera2Found:
    try:
        import picamera
        from picamera.array import PiRGBArray
        piCameraFound = True
        print("Camera.py: picamera v1 module available")
    except Exception as e:
        print("Camera.py: picamera v1 module not available:", e)
        piCameraFound = False

if not piCameraFound and not piCamera2Found:
    print("Camera.py: Using OpenCV interface (VideoCapture)")

def isRunningOnArlo():
    """Return True if we are running on a Pi camera or gstreamer pipeline."""
    return piCameraFound or piCamera2Found or gstreamerCameraFound

# OpenCV 3+ property helper
OPCV3 = int(cv2.__version__.split('.')[0]) >= 3
def capPropId(prop):
    return getattr(cv2 if OPCV3 else cv2.cv, ("" if OPCV3 else "CV_") + "CAP_PROP_" + prop)

def gstreamer_pipeline(capture_width=1280, capture_height=720, framerate=30):
    """A libcamera/gstreamer pipeline. Works on some setups; otherwise we fall back to plain VideoCapture."""
    return (
        "libcamerasrc ! "
        "videoconvert ! "
        f"video/x-raw, width=(int){capture_width}, height=(int){capture_height}, framerate=(fraction){framerate}/1 ! "
        "appsink"
    )

# --------------- ArUco setup ---------------
try:
    aruco = cv2.aruco
except Exception:
    raise RuntimeError("cv2.aruco not found. Install opencv-contrib-python (or -headless).")

ARUCO_DICT = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
ARUCO_PARAMS = aruco.DetectorParameters_create()

# --------------- Capture thread ---------------
class CaptureThread(threading.Thread):
    """Internal worker thread that captures frames from the camera and writes into a framebuffer."""
    def __init__(self, cam, backend, fb, picam_raw=None):
        super().__init__()
        self.cam = cam
        self.backend = backend  # 'picamera2' | 'picamera' | 'opencv'
        self.fb = fb
        self.picam_raw = picam_raw
        self._stop_evt = threading.Event()

    def stop(self):
        self._stop_evt.set()

    def run(self):
        while not self._stop_evt.is_set():
            if self.backend == "picamera":
                # Legacy picamera: capture directly into BGR array
                # Allocate buffer and capture (fast video port)
                if sys.version_info[0] > 2:
                    image = np.empty((self.cam.resolution[1], self.cam.resolution[0], 3), dtype=np.uint8)
                else:
                    image = np.empty((self.cam.resolution[1] * self.cam.resolution[0] * 3,), dtype=np.uint8)
                self.cam.capture(image, format="bgr", use_video_port=True)
                if sys.version_info[0] < 3:
                    image = image.reshape((self.cam.resolution[1], self.cam.resolution[0], 3))

            elif self.backend == "picamera2":
                # Picamera2 returns RGB; convert to BGR for consistency
                rgb = self.cam.capture_array()
                image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            else:  # 'opencv'
                ok, image = self.cam.read()
                if not ok:
                    print("CaptureThread: Could not read next frame")
                    break  # or continue; we break to avoid tight error loop

            self.fb.new_frame(image)

# --------------- Camera class ---------------
class Camera(object):
    """Camera wrapper with ArUco detection and multi-backend capture."""

    def __init__(self, camidx, robottype='arlo', useCaptureThread=False):
        """
        camidx: index for OpenCV VideoCapture
        robottype: 'arlo' | 'frindo' | 'scribbler' | 'macbookpro'
        useCaptureThread: whether to run capture in a worker thread with a framebuffer
        """
        print("robottype =", robottype)
        self.useCaptureThread = useCaptureThread
        self.FPS = 5  # target FPS (used for Picamera2 frame duration)

        # --- Set intrinsics by robot type ---
        if robottype == 'arlo':
            self.imageSize = (1640, 1232)
            self.intrinsic_matrix = np.asarray([
                1687.0, 0.,   self.imageSize[0] / 2.0,
                0.,    1687.0, self.imageSize[1] / 2.0,
                0.,    0.,    1.
            ], dtype=np.float64).reshape(3, 3)
            self.distortion_coeffs = np.asarray([0., 0., 2.0546093607192093e-02, -3.5538453075048249e-03, 0.], dtype=np.float64)

        elif robottype == 'frindo':
            self.imageSize = (640, 480)
            self.intrinsic_matrix = np.asarray([
                500., 0.,   self.imageSize[0] / 2.0,
                0.,  500.,  self.imageSize[1] / 2.0,
                0.,  0.,    1.
            ], dtype=np.float64).reshape(3, 3)
            self.distortion_coeffs = np.asarray([0., 0., 2.0546093607192093e-02, -3.5538453075048249e-03, 0.], dtype=np.float64)

        elif robottype == 'scribbler':
            self.imageSize = (640, 480)
            self.intrinsic_matrix = np.asarray([
                713.05391967046853, 0., 311.72820723774367,
                0., 705.64929862291285, 256.34470978315028,
                0., 0., 1.
            ], dtype=np.float64).reshape(3, 3)
            self.distortion_coeffs = np.asarray([0.11911006, -1.00033662, 0.01928790, -0.00237282, -0.28137266], dtype=np.float64)

        elif robottype == 'macbookpro':
            self.imageSize = (1280, 720)
            self.intrinsic_matrix = np.asarray([
                943.2809516292071, 0.,   self.imageSize[0] / 2.0,
                0.,  949.4666859597908,  self.imageSize[1] / 2.0,
                0.,  0.,  1.
            ], dtype=np.float64).reshape(3, 3)
            self.distortion_coeffs = np.asarray([0., 0., -0.016169374082976234, 0.008765765317006246, 0.], dtype=np.float64)

        else:
            print("Camera.__init__: Unknown robot type")
            sys.exit(-1)

        # ---- Open backend ----
        self.backend = None
        self.cam = None
        self.rawCapture = None  # for picamera v1

        W, H = self.imageSize

        if piCamera2Found:
            self.backend = "picamera2"
            self.cam = Picamera2()
            frame_duration_us = int(1.0 / self.FPS * 1_000_000)
            cfg = self.cam.create_video_configuration(
                main={"size": (W, H), "format": "RGB888"},
                controls={"FrameDurationLimits": (frame_duration_us, frame_duration_us)},
                queue=False
            )
            self.cam.configure(cfg)
            self.cam.start(show_preview=False)
            time.sleep(0.5)

        elif piCameraFound:
            self.backend = "picamera"
            self.cam = picamera.PiCamera(camera_num=camidx, resolution=self.imageSize, framerate=30)
            if not self.useCaptureThread:
                self.rawCapture = PiRGBArray(self.cam, size=self.cam.resolution)
            time.sleep(2)  # warm-up
            # Lock WB/exposure for stability
            gain = self.cam.awb_gains
            self.cam.awb_mode = 'off'
            self.cam.awb_gains = gain
            self.cam.shutter_speed = self.cam.exposure_speed
            self.cam.exposure_mode = 'off'
            print("shutter_speed =", self.cam.shutter_speed)
            print("awb_gains     =", self.cam.awb_gains)
            print("Camera width  =", self.cam.resolution[0])
            print("Camera height =", self.cam.resolution[1])
            print("Camera FPS    =", self.cam.framerate)

        else:
            # Try gstreamer first
            pipeline = gstreamer_pipeline(capture_width=W, capture_height=H, framerate=30)
            cap = cv2.VideoCapture(pipeline, apiPreference=cv2.CAP_GSTREAMER)
            if cap.isOpened():
                self.backend = "opencv"
                self.cam = cap
                gstreamerCameraFound = True
                print("Camera.__init__: Using OpenCV with gstreamer")
            else:
                # Fallback to auto-detect backend
                self.cam = cv2.VideoCapture(camidx)
                if not self.cam.isOpened():
                    print("Camera.__init__: Could not open camera")
                    sys.exit(-1)
                self.backend = "opencv"
                print("Camera.__init__: Using OpenCV auto-detect")

            time.sleep(1)
            self.cam.set(capPropId("FRAME_WIDTH"), W)
            self.cam.set(capPropId("FRAME_HEIGHT"), H)
            self.cam.set(capPropId("FPS"), 30)
            time.sleep(0.5)
            print("Camera width  =", int(self.cam.get(capPropId("FRAME_WIDTH"))))
            print("Camera height =", int(self.cam.get(capPropId("FRAME_HEIGHT"))))
            print("Camera FPS    =", int(self.cam.get(capPropId("FPS"))))

        # ---- Distortion maps (disabled; we use cv2.undistort when/if needed) ----
        # self.mapx, self.mapy = cv2.initUndistortRectifyMap(...)

        # ---- Chessboard params (kept from your original code) ----
        self.patternFound = False
        self.patternSize = (3, 4)
        self.patternUnit = 50.0  # mm
        self.corners = []

        # ---- ArUco dictionary + marker size (meters) ----
        self.arucoDict = ARUCO_DICT
        self.arucoParams = ARUCO_PARAMS
        self.arucoMarkerLength = 0.15  # meters

        # ---- Optional capture thread + framebuffer ----
        self.framebuffer = None
        self.capturethread = None
        if self.useCaptureThread:
            if framebuffer is None:
                raise RuntimeError("framebuffer module not available but useCaptureThread=True")
            print("Using capture thread")
            self.framebuffer = framebuffer.FrameBuffer((self.imageSize[1], self.imageSize[0], 3))
            self.capturethread = CaptureThread(
                cam=self.cam,
                backend=self.backend,
                fb=self.framebuffer,
                picam_raw=self.rawCapture
            )
            self.capturethread.start()
            time.sleep(0.75)

        # placeholders for last detection (used by draw_aruco_objects)
        self.aruco_corners = None
        self.ids = None
        self.rvecs = None
        self.tvecs = None

    def __del__(self):
        # Best-effort cleanup
        try:
            if self.capturethread is not None:
                self.capturethread.stop()
                self.capturethread.join()
        except Exception:
            pass
        try:
            if self.backend == "picamera2" and self.cam is not None:
                self.cam.stop()
            elif self.backend == "picamera" and self.cam is not None:
                self.cam.close()
            elif self.backend == "opencv" and self.cam is not None:
                self.cam.release()
        except Exception:
            pass

    def terminateCaptureThread(self):
        if self.capturethread is not None:
            self.capturethread.stop()
            self.capturethread.join()
            self.capturethread = None

    def get_capture(self):
        """Direct access to underlying camera object."""
        return self.cam

    # Back-compat alias
    def get_colour(self):
        print("OBSOLETE get_colour - use get_next_frame() instead")
        return self.get_next_frame()

    def get_next_frame(self):
        """Get next BGR frame as np.uint8 (H, W, 3)."""
        if self.useCaptureThread:
            img = self.framebuffer.get_frame()
            if img is None:
                # Fallback to a black frame if buffer empty
                img = np.zeros((self.imageSize[1], self.imageSize[0], 3), dtype=np.uint8)
            return img

        # Non-threaded paths
        if self.backend == "picamera":
            self.rawCapture.truncate(0)
            self.cam.capture(self.rawCapture, format="bgr", use_video_port=True)
            return self.rawCapture.array  # already BGR

        elif self.backend == "picamera2":
            rgb = self.cam.capture_array()
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        else:  # opencv
            ok, img = self.cam.read()
            if not ok:
                print("Camera.get_next_frame: Could not read next frame")
                sys.exit(-1)
            return img  # BGR

    # --------------- ArUco ---------------
    def detect_aruco_objects(self, img_bgr):
        """
        Detect ArUco markers and return:
          ids: np.ndarray shape (K,) or None
          dists_cm: np.ndarray shape (K,) distances to marker centers [cm]
          angles_rad: np.ndarray shape (K,) bearings (signed, rad)
        Distance/bearing computed from cv2.aruco pose estimation using camera intrinsics.
        """
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        self.aruco_corners, self.ids, _rej = aruco.detectMarkers(gray, self.arucoDict, parameters=self.arucoParams)

        self.rvecs, self.tvecs = None, None
        if self.ids is None or len(self.aruco_corners) == 0:
            return None, None, None

        self.rvecs, self.tvecs, _objPoints = aruco.estimatePoseSingleMarkers(
            self.aruco_corners, self.arucoMarkerLength, self.intrinsic_matrix, self.distortion_coeffs
        )

        # Distances (cm) and horizontal bearings (rad)
        tvecs = self.tvecs.reshape(-1, 3)  # meters in camera frame
        dists = (np.linalg.norm(tvecs, axis=1) * 100.0)  # to cm

        # Bearing: project onto x-z plane; angle from +z (forward) with sign by x
        angles = np.zeros(dists.shape, dtype=np.float64)
        for i in range(tvecs.shape[0]):
            x, y, z = tvecs[i]
            if z == 0.0:
                angles[i] = 0.0
                continue
            # angle positive to the LEFT (x>0) when facing +z
            # atan2 uses (y, x); here we want angle in x–z plane from +z:
            # theta = atan2(x, z)
            angles[i] = np.arctan2(x, z)

        ids = self.ids.reshape(-1)
        return ids, dists, angles

    def draw_aruco_objects(self, img_bgr):
        """Draw detected markers and axes on the BGR image (in-place)."""
        if self.ids is None or self.aruco_corners is None:
            return
        aruco.drawDetectedMarkers(img_bgr, self.aruco_corners, self.ids)
        if self.rvecs is not None and self.tvecs is not None:
            for i in range(self.ids.shape[0]):
                cv2.drawFrameAxes(img_bgr, self.intrinsic_matrix, self.distortion_coeffs,
                                  self.rvecs[i], self.tvecs[i], self.arucoMarkerLength)

    # --------------- Chessboard (kept from original) ---------------
    def get_object(self, img_bgr):
        """Detect chessboard; return (type, distance_cm, angle_rad, colourProb RGB)."""
        objectType = 'none'
        colourProb = np.ones((3,)) / 3.0
        distance = 0.0
        angle = 0.0
        self.patternFound = False

        self.get_corners(img_bgr)
        if self.patternFound:
            delta_x = abs(self.corners[0, 0, 0] - self.corners[2, 0, 0])
            delta_y = abs(self.corners[0, 0, 1] - self.corners[2, 0, 1])
            horizontal = (delta_y > delta_x)
            objectType = 'horizontal' if horizontal else 'vertical'

            if horizontal:
                height = ((abs(self.corners[0, 0, 1] - self.corners[2, 0, 1]) +
                           abs(self.corners[9, 0, 1] - self.corners[11, 0, 1])) / 2.0)
                patternHeight = (self.patternSize[0] - 1.0) * self.patternUnit
            else:
                height = (abs(self.corners[0, 0, 1] - self.corners[9, 0, 1]) +
                          abs(self.corners[2, 0, 1] - self.corners[11, 0, 1])) / 2.0
                patternHeight = (self.patternSize[1] - 1.0) * self.patternUnit

            distance = self.intrinsic_matrix[1, 1] * patternHeight / (height * 10.0)  # to cm
            center = (self.corners[0, 0, 0] + self.corners[2, 0, 0] +
                      self.corners[9, 0, 0] + self.corners[11, 0, 0]) / 4.0
            angle = -np.arctan2(center - self.intrinsic_matrix[0, 2], self.intrinsic_matrix[0, 0])

            # Colour classification inside polygon (simple mean)
            points = np.array([self.corners[0], self.corners[2], self.corners[9], self.corners[11]])
            points.shape = (4, 2)
            points = np.int32(points)

            mask = np.zeros((self.imageSize[1], self.imageSize[0]), dtype=np.uint8)
            cv2.fillConvexPoly(mask, points, 255)
            mean_colour = cv2.mean(img_bgr, mask=mask)  # B,G,R,A
            b, g, r = mean_colour[:3]
            s = r + g + b if (r + g + b) != 0 else 1.0
            colourProb = np.array([r / s, g / s, b / s])

        return objectType, distance, angle, colourProb

    def get_corners(self, img_bgr):
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        loggray = cv2.log(gray + 1.0)
        cv2.normalize(loggray, loggray, 0, 255, cv2.NORM_MINMAX)
        gray = cv2.convertScaleAbs(loggray)
        retval, self.corners = cv2.findChessboardCorners(
            gray, self.patternSize, cv2.CALIB_CB_FAST_CHECK
        )
        self.patternFound = bool(retval)
        return self.patternFound, self.corners

    def draw_object(self, img_bgr):
        cv2.drawChessboardCorners(img_bgr, self.patternSize, self.corners, self.patternFound)


# --------------- Standalone test ---------------
if __name__ == '__main__':
    print("Opening and initializing camera")
    cam = Camera(0, robottype='arlo', useCaptureThread=False)

    WIN_RF1 = "Camera view"
    cv2.namedWindow(WIN_RF1)
    cv2.moveWindow(WIN_RF1, 50, 50)

    while True:
        action = cv2.waitKey(10)
        if action == ord('q'):
            break

        colour = cam.get_next_frame()  # BGR

        IDs, dists, angles = cam.detect_aruco_objects(colour)
        if IDs is not None:
            for i in range(len(IDs)):
                print(f"Object ID = {IDs[i]}, Distance = {dists[i]:.1f} cm, angle = {angles[i]:.3f} rad")

        cam.draw_aruco_objects(colour)
        cv2.imshow(WIN_RF1, colour)

    cv2.destroyAllWindows()
    cam.terminateCaptureThread()
