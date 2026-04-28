import time
import cv2
import sys
from camera_core import CameraManager
from ocr_worker import OCRWorker


def main():
    # Allow confidence threshold via command line (default 0.0 = no filtering)
    min_conf = 0.0
    if len(sys.argv) > 1:
        try:
            min_conf = float(sys.argv[1])
            print(f"Using confidence threshold: {min_conf*100:.0f}%")
        except ValueError:
            print(f"Invalid threshold, using default 0% (no filtering)")

    print("Initializing OCR Engine...")
    ocr = OCRWorker(lang='en', min_confidence=min_conf)

    if ocr.ocr_engine is None:
        print("Failed to load OCR engine. Please check dependencies.")
        return

    print("\nInitializing Camera...")
    cam_manager = CameraManager()

    try:
        cam_manager.initialize_camera()
        time.sleep(2)
        print("Camera ready. Position text in front of the camera and press SPACE to capture.")
        print("Diagnostic pixel values printed above — share them if colors are still wrong.")

        captured_frame = None

        while True:
            # Use lores stream for preview (matches reference code)
            frame = cam_manager.get_preview_frame()
            if frame is None:
                continue

            cv2.imshow("Camera Feed - SPACE: capture | Q: quit", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("Exiting.")
                return
            elif key == 32:
                captured_frame = cam_manager.get_frame()
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
        ocr.extract_text(captured_frame, preprocess=True)


if __name__ == "__main__":
    main()
