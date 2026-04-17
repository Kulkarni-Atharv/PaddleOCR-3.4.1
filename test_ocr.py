import sys
import time
import cv2
from camera_core import CameraManager
from ocr_worker import OCRWorker

def main():
    print("Initializing OCR Engine...")
    ocr = OCRWorker(use_gpu=False, lang='en')
    
    if ocr.ocr_engine is None:
        print("Failed to load OCR engine. Please check dependencies.")
        return

    print("\nInitializing Camera...")
    cam_manager = CameraManager()
    
    try:
        cam_manager.initialize_camera()
        time.sleep(2) # Give the camera sensor a moment to warm up
        print("Camera ready. Opening Live Feed...")
        
        while True:
            # Capture frame continuously for live preview
            frame = cam_manager.get_frame()
            if frame is None:
                continue
                
            # Resize the frame so it fits nicely on the screen for the live preview
            preview_frame = cv2.resize(frame, (640, 480))
            
            # Display the video window
            cv2.imshow("Camera Feed - Press SPACE to extract text, Q to quit", preview_frame)
            
            # Wait for 1ms and check for key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("Exiting...")
                break
            elif key == 32: # Spacebar key code
                print("\nSpacebar pressed! Extracting text...")
                # We pass the full, un-resized frame to the OCR worker for maximum detail
                ocr.extract_text(frame, preprocess=True)
                
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
