from picamera2 import Picamera2
import cv2


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
                "AwbMode": 1,
            }
        )
        self.picam2.configure(video_config)
        self.picam2.start()

    def get_frame(self):
        if not self.picam2:
            return None
        try:
            # RGB888 format: frame is already in RGB — return as-is.
            # Callers that use cv2.imshow must convert to BGR themselves.
            return self.picam2.capture_array("main")
        except Exception as e:
            print("Error capturing frame:", e)
            return None

    def close(self):
        if self.picam2:
            self.picam2.stop()
