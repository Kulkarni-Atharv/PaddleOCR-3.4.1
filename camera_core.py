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
            main={"size": (1456, 1088), "format": "RGB888"},
            lores={"size": (640, 480), "format": "RGB888"},
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
        """Captures a frame and returns it as a standard OpenCV BGR array."""
        if not self.picam2:
            return None
        try:
            # Capture frame
            frame = self.picam2.capture_array("main")
            
            # If the image looks "blue to orange", it means the Red and Blue channels are swapped.
            # This directly swaps the channels to fix the inversion.
            corrected_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # If it STILL looks blue to orange, change the above line to:
            # corrected_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            return corrected_frame
        except Exception as e:
            print("Error capturing frame:", e)
            return None
