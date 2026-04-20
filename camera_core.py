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
            return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        except Exception as e:
            print("Error capturing frame:", e)
            return None

    def close(self):
        if self.picam2:
            self.picam2.stop()
