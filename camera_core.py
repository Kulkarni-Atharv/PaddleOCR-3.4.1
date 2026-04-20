from picamera2 import Picamera2
import cv2

class CameraManager:
    def __init__(self, exposure=10000, gain=4.0, fps=30):
        self.exposure = exposure
        self.gain = gain
        self.fps = fps
        self.picam2 = None
        self._diag_done = False

    def initialize_camera(self):
        self.picam2 = Picamera2()

        # XBGR8888 is the native ISP output format for video on Raspberry Pi.
        # Requesting BGR888/RGB888 in video mode may be silently ignored, causing
        # the infamous orange↔blue swap. Using XBGR8888 + BGRA2BGR is reliable.
        video_config = self.picam2.create_video_configuration(
            main={"size": (1456, 1088), "format": "XBGR8888"},
            lores={"size": (640, 480), "format": "XBGR8888"},
            controls={
                "FrameRate": self.fps,
                "ExposureTime": self.exposure,
                "AnalogueGain": self.gain,
                "AwbMode": 0,
            }
        )
        self.picam2.configure(video_config)
        self.picam2.start()

    def get_frame(self):
        if not self.picam2:
            return None
        try:
            frame = self.picam2.capture_array("main")

            # One-time diagnostic: confirms actual format coming from camera
            if not self._diag_done:
                self._diag_done = True
                h, w = frame.shape[:2]
                print(f"[CameraManager] shape={frame.shape} dtype={frame.dtype} "
                      f"center_pixel={frame[h//2, w//2]}")

            # XBGR8888 → channels [B, G, R, X]; drop X, keep BGR for OpenCV
            if frame.ndim == 3 and frame.shape[2] == 4:
                return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            # Fallback: 3-channel RGB (shouldn't happen with XBGR8888, but safe)
            return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        except Exception as e:
            print("Error capturing frame:", e)
            return None

    def close(self):
        if self.picam2:
            self.picam2.stop()
