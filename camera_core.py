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

    def get_frame(self):
        if not self.picam2:
            return None
        try:
            # PiCamera2 RGB888 returns BGR byte order in numpy.
            # cv2.imshow consumers: use directly (expects BGR).
            # PIL/Tkinter consumers: apply cv2.COLOR_BGR2RGB before Image.fromarray.
            return self.picam2.capture_array("main")
        except Exception as e:
            print("Error capturing frame:", e)
            return None

    def close(self):
        if self.picam2:
            self.picam2.stop()
