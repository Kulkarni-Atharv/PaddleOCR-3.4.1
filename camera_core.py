from picamera2 import Picamera2


class CameraManager:
    def __init__(self, exposure=10000, gain=4.0, fps=30):
        self.exposure = exposure
        self.gain = gain
        self.fps = fps
        self.picam2 = None

    def initialize_camera(self):
        self.picam2 = Picamera2()

        video_config = self.picam2.create_video_configuration(
            main={"size": (1456, 1088), "format": "RGB888"},
            lores={"size": (640, 480), "format": "RGB888"},
            controls={
                "FrameRate": self.fps,
                "ExposureTime": self.exposure,
                "AnalogueGain": self.gain,
            }
        )
        self.picam2.configure(video_config)
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
            return self.picam2.capture_array("main")
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
