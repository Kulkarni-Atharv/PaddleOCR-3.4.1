import cv2
from picamera2 import Picamera2


class CameraManager:
    def __init__(self):
        self.picam2 = None

    def initialize_camera(self):
        self.picam2 = Picamera2()

        config = self.picam2.create_preview_configuration(
            main={"size": (1456, 1088)},
            lores={"size": (640, 480), "format": "BGR888"},
            display="main"
        )
        self.picam2.configure(config)
        self.picam2.start()

        # Diagnostic: print actual pixel layout on first capture
        test = self.picam2.capture_array("main")
        h, w = test.shape[:2]
        print(f"[Camera] main stream: shape={test.shape} dtype={test.dtype}")
        print(f"[Camera] center pixel channels: {test[h//2, w//2]}")
        test2 = self.picam2.capture_array("lores")
        print(f"[Camera] lores stream: shape={test2.shape} dtype={test2.dtype}")
        print(f"[Camera] center pixel channels: {test2[h//2//2, w//2//2]}")

    def get_frame(self):
        """High-res main stream for OCR (1456x1088)."""
        if not self.picam2:
            return None
        try:
            frame = self.picam2.capture_array("main")
            return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        except Exception as e:
            print("Error capturing frame:", e)
            return None

    def get_preview_frame(self):
        """Low-res lores stream for live preview (640x480)."""
        if not self.picam2:
            return None
        try:
            return self.picam2.capture_array("lores")
        except Exception as e:
            print("Error capturing preview frame:", e)
            return None

    def close(self):
        if self.picam2:
            self.picam2.stop()
