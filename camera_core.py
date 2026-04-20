from picamera2 import Picamera2
import cv2
import time

class CameraManager:
    def __init__(self, exposure=15000, gain=4.0, fps=30):
        self.exposure = exposure
        self.gain = gain
        self.fps = fps
        self.picam2 = None

    def initialize_camera(self):
        self.picam2 = Picamera2()
        
        # Use standard RGB format
        video_config = self.picam2.create_video_configuration(
            main={"size": (1456, 1088), "format": "BGR888"},
            lores={"size": (640, 480), "format": "BGR888"},
            controls={
                "FrameRate": self.fps,
                "ExposureTime": self.exposure,
                "AnalogueGain": self.gain,
                "AwbMode": 1  # 1 = Auto White Balance (fixes weird coloring)
            }
        )
        self.picam2.configure(video_config)
        self.picam2.start()

    def close(self):
        if self.picam2:
            self.picam2.stop()

def get_frame(self):
    """Captures a frame and returns it correctly for OpenCV."""
    if not self.picam2:
        return None
    try:
        
        frame = self.picam2.capture_array("main")

        return frame

    except Exception as e:
        print("Error capturing frame:", e)
        return None
