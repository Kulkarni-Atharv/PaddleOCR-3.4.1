import cv2
import numpy as np
import logging
import os

# ── ARM64 / CM5 stability ─────────────────────────────────────────────────────
# All of these must be set BEFORE paddle / paddleocr are imported.
# The segfault on ARM64 is caused by multi-threaded NEON/SIMD ops in PaddlePaddle.
# Forcing every threading layer to 1 thread eliminates the crash.
os.environ["OPENBLAS_NUM_THREADS"]         = "1"
os.environ["OMP_NUM_THREADS"]              = "1"
os.environ["MKL_NUM_THREADS"]              = "1"
os.environ["BLIS_NUM_THREADS"]             = "1"
os.environ["FLAGS_paddle_num_threads"]     = "1"
os.environ["FLAGS_use_mkldnn"]             = "0"
os.environ["FLAGS_logtostderr"]            = "0"
os.environ["FLAGS_minloglevel"]            = "3"
os.environ["PADDLE_CPP_MAIN_LOG_LEVEL"]    = "2"
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
# ─────────────────────────────────────────────────────────────────────────────

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
for _log in ('ppocr', 'paddleocr', 'paddlex', 'paddle', 'root'):
    logging.getLogger(_log).setLevel(logging.ERROR)


class OCRWorker:
    """
    Text extraction using PaddleOCR 3.x (PP-OCRv5 models) on CM5 (ARM64).
    Single-threaded inference prevents the NEON segfault on ARM64.
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
                cpu_threads=1,        # 1 thread — prevents ARM64 NEON segfault
                det_limit_side_len=960,
                det_db_thresh=0.3,
                det_db_box_thresh=0.5,
                rec_batch_num=1,      # 1 batch — reduces memory pressure on CM5
            )
            logging.info("PaddleOCR initialized successfully.")
        except Exception as e:
            logging.error(f"Failed to initialize PaddleOCR: {e}")

    def preprocess_image(self, image):
        if image is None:
            return None

        resized = cv2.resize(image, (1280, 960), interpolation=cv2.INTER_LANCZOS4)

        # CLAHE for better local contrast on faint/low-contrast text
        lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_eq = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
        enhanced = cv2.cvtColor(cv2.merge([l_eq, a, b]), cv2.COLOR_LAB2BGR)

        # Unsharp mask to crisp up micro-fonts
        blur = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=1.5)
        return cv2.addWeighted(enhanced, 1.6, blur, -0.6, 0)

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
            # PaddleOCR 3.x predict() — result is a list, one entry per image.
            # Each entry is a dict (or object) with rec_texts / rec_scores.
            results = self.ocr_engine.predict(process_img)

            extracted_text = []
            if results:
                for res in results:
                    if isinstance(res, dict):
                        texts  = res.get('rec_texts', [])
                        scores = res.get('rec_scores', [])
                    else:
                        texts  = getattr(res, 'rec_texts', [])
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
