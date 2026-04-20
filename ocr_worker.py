import cv2
import numpy as np
import logging
import os

# Set before importing PaddlePaddle — ARM thread safety + log suppression
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["PADDLE_CPP_MAIN_LOG_LEVEL"] = "2"
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
os.environ["FLAGS_logtostderr"] = "0"
os.environ["FLAGS_minloglevel"] = "3"  # Suppress PaddlePaddle C++ INFO/WARNING

try:
    from paddleocr import PaddleOCR
except ImportError:
    logging.warning(
        "PaddleOCR not installed. Run:\n"
        "  pip install paddlepaddle\n"
        "  pip install paddleocr==3.4.1"
    )
    PaddleOCR = None

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Suppress PaddleOCR / PaddleX internal loggers
for _name in ('ppocr', 'paddleocr', 'paddlex', 'paddle', 'root'):
    logging.getLogger(_name).setLevel(logging.ERROR)


class OCRWorker:
    """
    Text extraction using PaddleOCR 3.x (PP-OCRv5 models).
    Tuned for micro-level font detection on CM5 (ARM64) with a global shutter camera.
    """
    def __init__(self, lang='en'):
        self.lang = lang
        self.ocr_engine = None
        self._initialize_engine()

    def _initialize_engine(self):
        if PaddleOCR is None:
            logging.error("Cannot initialize: paddleocr is missing.")
            return

        logging.info("Initializing PaddleOCR (PP-OCRv5)...")
        try:
            self.ocr_engine = PaddleOCR(
                use_angle_cls=True,
                lang=self.lang,
                device='cpu',
                cpu_threads=4,
                det_limit_side_len=1280,
                det_db_thresh=0.3,
                det_db_box_thresh=0.5,
                rec_batch_num=6,
            )
            logging.info("PaddleOCR initialized successfully.")
        except Exception as e:
            logging.error(f"Failed to initialize PaddleOCR: {e}")

    def preprocess_image(self, image):
        if image is None:
            return None

        # 1280x960 gives PaddleOCR enough detail for fine text; CM5 handles it fine
        resized = cv2.resize(image, (1280, 960), interpolation=cv2.INTER_LANCZOS4)

        # CLAHE on L channel for better local contrast on low-contrast/faint text
        lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_eq = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
        enhanced = cv2.cvtColor(cv2.merge([l_eq, a, b]), cv2.COLOR_LAB2BGR)

        # Unsharp mask to crisp up micro-fonts
        blur = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=1.5)
        sharpened = cv2.addWeighted(enhanced, 1.6, blur, -0.6, 0)

        return sharpened

    def extract_text(self, image, preprocess=True):
        """
        Extracts text from the provided image frame.

        Args:
            image:      numpy array — BGR image from OpenCV/Camera
            preprocess: bool — whether to apply micro-font preprocessing

        Returns:
            str: Detected text lines joined by newlines.
        """
        if self.ocr_engine is None:
            logging.error("OCR engine is not loaded.")
            return ""

        if image is None:
            logging.warning("Image is None. Skipping OCR.")
            return ""

        process_img = self.preprocess_image(image) if preprocess else image

        logging.info("Starting text extraction...")
        try:
            # PaddleOCR 3.x predict() returns a list of result objects (one per image).
            # Each result exposes rec_texts and rec_scores either as dict keys or attributes.
            results = self.ocr_engine.predict(process_img)

            extracted_text = []
            if results:
                for res in results:
                    # Handle both dict-style and attribute-style access
                    if isinstance(res, dict):
                        texts = res.get('rec_texts', [])
                        scores = res.get('rec_scores', [])
                    else:
                        texts = getattr(res, 'rec_texts', [])
                        scores = getattr(res, 'rec_scores', [])

                    for text, score in zip(texts, scores):
                        try:
                            confidence = float(score)
                            extracted_text.append(text)
                            logging.info(f"Detected: '{text}' (Confidence: {confidence:.2f})")
                        except (TypeError, ValueError):
                            extracted_text.append(str(text))

            final_text = "\n".join(extracted_text)

            if not final_text:
                logging.info("No text detected.")
            else:
                print("\n--- OCR EXTRACTION RESULT ---")
                print(final_text)
                print("-----------------------------\n")

            return final_text

        except Exception as e:
            logging.error(f"Error during OCR extraction: {e}")
            return ""
