import time
import threading
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

    # Prevents overlapping OCR calls and signals the preview overlay
    ocr_running = threading.Event()

    def run_ocr(frame):
        try:
            ocr.extract_text(frame, preprocess=True)
        finally:
            ocr_running.clear()

    try:
        cam_manager.initialize_camera()
        time.sleep(2)
        print("Camera ready. Opening Live Feed...")

        while True:
            frame = cam_manager.get_frame()
            if frame is None:
                continue

            preview = cv2.resize(frame, (640, 480))

            if ocr_running.is_set():
                cv2.putText(preview, "OCR Processing...", (10, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

            cv2.imshow("Camera Feed - SPACE: extract text | Q: quit", preview)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("Exiting...")
                break
            elif key == 32:
                if ocr_running.is_set():
                    print("OCR still running — please wait.")
                else:
                    print("\nSpacebar pressed! Extracting text...")
                    ocr_running.set()
                    # Copy frame so the camera loop can keep overwriting `frame`
                    t = threading.Thread(target=run_ocr, args=(frame.copy(),), daemon=True)
                    t.start()

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        print("Cleaning up resources...")
        cam_manager.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
