import cv2
import numpy as np
import logging
import os

# Must be set before importing PaddlePaddle — prevents ARM thread crashes on CM5
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["PADDLE_CPP_MAIN_LOG_LEVEL"] = "2"
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"  # Skip slow connectivity check on startup

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
# Suppress PaddleOCR's own verbose loggers — show_log param was removed in 3.x
logging.getLogger('ppocr').setLevel(logging.WARNING)
logging.getLogger('paddleocr').setLevel(logging.WARNING)


class OCRWorker:
    """
    Text extraction using PaddleOCR 3.4.1 (PP-OCRv5 models).
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

        logging.info("Initializing PaddleOCR 3.4.1 (PP-OCRv5)...")
        try:
            self.ocr_engine = PaddleOCR(
                use_angle_cls=True,       # Detect rotated/upside-down text
                lang=self.lang,
                use_gpu=False,
                enable_mkldnn=False,      # Intel MKL-DNN — must be False on ARM64
                cpu_threads=4,            # CM5 has 4 cores
                det_limit_side_len=1280,  # Max side length for detector; higher = finer text
                det_db_thresh=0.3,        # Lower threshold catches faint/thin text edges
                det_db_box_thresh=0.5,    # Min score to keep a detected box
                rec_batch_num=6,
            )
            logging.info("PaddleOCR 3.4.1 initialized successfully.")
        except Exception as e:
            logging.error(f"Failed to initialize PaddleOCR: {e}")

    def preprocess_image(self, image):
        """
        Enhances micro-level fonts before handing off to PaddleOCR.
        PaddleOCR does its own internal preprocessing on top of this.
        """
        if image is None:
            return None

        # CM5 has enough RAM for 1280x960 — gives PaddleOCR more detail on fine text
        resized = cv2.resize(image, (1280, 960), interpolation=cv2.INTER_LANCZOS4)

        # CLAHE on the L channel (LAB space) for better local contrast on low-contrast text
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
            logging.error("OCR engine is not loaded. Cannot extract text.")
            return ""

        if image is None:
            logging.warning("Provided image is None. Skipping OCR.")
            return ""

        process_img = self.preprocess_image(image) if preprocess else image

        logging.info("Starting text extraction...")
        try:
            # PaddleOCR 3.x API: predict() returns a list of result dicts, one per input image.
            # Each dict contains:
            #   'rec_texts'  — list of recognized text strings
            #   'rec_scores' — list of corresponding confidence floats
            #   'dt_polys'   — list of bounding polygons (not used here)
            results = self.ocr_engine.predict(process_img)

            extracted_text = []
            if results:
                for res in results:
                    texts = res.get('rec_texts', [])
                    scores = res.get('rec_scores', [])
                    for text, score in zip(texts, scores):
                        try:
                            confidence = float(score)
                            extracted_text.append(text)
                            logging.info(f"Detected: '{text}' (Confidence: {confidence:.2f})")
                        except (TypeError, ValueError):
                            extracted_text.append(text)

            final_text = "\n".join(extracted_text)

            if not final_text:
                logging.info("No text detected in the image.")
            else:
                print("\n--- OCR EXTRACTION RESULT ---")
                print(final_text)
                print("-----------------------------\n")

            return final_text

        except Exception as e:
            logging.error(f"Error during OCR extraction: {e}")
            return ""
