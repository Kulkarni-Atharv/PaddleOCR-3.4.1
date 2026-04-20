import time
import cv2
from camera_core import CameraManager
from ocr_worker import OCRWorker


def main():
    print("Initializing OCR Engine...")
    ocr = OCRWorker(lang='en')

    if ocr.ocr_engine is None:
        print("Failed to load OCR engine. Please check dependencies.")
        return

    print("\nInitializing Camera...")
    cam_manager = CameraManager()

    try:
        cam_manager.initialize_camera()
        time.sleep(2)
        print("Camera ready. Position text in front of the camera and press SPACE to capture.")

        captured_frame = None

        while True:
            frame = cam_manager.get_frame()
            if frame is None:
                continue

            # frame is RGB; cv2.imshow expects BGR
            preview = cv2.cvtColor(cv2.resize(frame, (640, 480)), cv2.COLOR_RGB2BGR)
            cv2.imshow("Camera Feed - SPACE: capture | Q: quit", preview)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("Exiting.")
                return
            elif key == 32:
                captured_frame = frame.copy()
                print("Image captured. Closing preview...")
                break

    except KeyboardInterrupt:
        print("\nInterrupted.")
        return
    finally:
        cam_manager.close()
        cv2.destroyAllWindows()

    if captured_frame is not None:
        print("Processing image...\n")
        # OCR pipeline (OpenCV internally) expects BGR
        ocr.extract_text(cv2.cvtColor(captured_frame, cv2.COLOR_RGB2BGR), preprocess=True)


if __name__ == "__main__":
    main()
