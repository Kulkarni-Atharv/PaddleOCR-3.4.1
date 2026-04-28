import cv2
import gc
import numpy as np
import logging
import os

# ── ARM64 / CM5 stability — must be set before paddle imports ─────────────────
os.environ["OPENBLAS_NUM_THREADS"]              = "1"
os.environ["OMP_NUM_THREADS"]                   = "1"
os.environ["MKL_NUM_THREADS"]                   = "1"
os.environ["BLIS_NUM_THREADS"]                  = "1"
os.environ["FLAGS_paddle_num_threads"]          = "1"
os.environ["FLAGS_use_mkldnn"]                  = "0"
os.environ["FLAGS_logtostderr"]                 = "0"
os.environ["FLAGS_minloglevel"]                 = "3"
os.environ["PADDLE_CPP_MAIN_LOG_LEVEL"]         = "2"
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
# ─────────────────────────────────────────────────────────────────────────────

try:
    from paddleocr import PaddleOCR
except ImportError:
    logging.warning("PaddleOCR not installed. Run: pip install paddlepaddle paddleocr==3.4.1")
    PaddleOCR = None

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
for _log in ('ppocr', 'paddleocr', 'paddlex', 'paddle', 'root'):
    logging.getLogger(_log).setLevel(logging.ERROR)


class OCRWorker:
    """
    PaddleOCR 3.x on CM5 (ARM64, 4 GB RAM).
    Only the det + rec + angle models are loaded; doc-orientation and UVDoc
    are disabled to save ~120 MB of peak RAM.
    """
    def __init__(self, lang='en', min_confidence=0.90):
        self.lang = lang
        self.min_confidence = min_confidence  # Filter words below this confidence
        self.ocr_engine = None
        self._initialize_engine()

    def _initialize_engine(self):
        if PaddleOCR is None:
            logging.error("Cannot initialize: paddleocr is missing.")
            return

        logging.info("Initializing PaddleOCR (PP-OCRv5)...")
        try:
            self.ocr_engine = PaddleOCR(
                lang=self.lang,
                device='cpu',
                cpu_threads=1,                    # ARM64: multi-thread causes NEON segfault
                use_angle_cls=True,
                use_doc_orientation_classify=False, # Saves ~20 MB, not needed for live camera
                use_doc_unwarping=False,            # Saves ~100 MB, not needed for camera feed
                det_limit_side_len=640,            # Caps det input; keeps peak inference RAM low
                det_db_thresh=0.5,                 # 50% - Stricter detection
                det_db_box_thresh=0.6,             # 60% - Stricter box filtering
                rec_batch_num=1,
            )
            logging.info("PaddleOCR initialized.")
        except Exception as e:
            logging.error(f"Failed to initialize PaddleOCR: {e}")

    def preprocess_image(self, image):
        if image is None:
            return None

        # 800x600 matches what PaddleOCR will internally scale to at det_limit_side_len=640
        resized = cv2.resize(image, (800, 600), interpolation=cv2.INTER_LANCZOS4)

        # Convert to grayscale for noise reduction
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        # Apply bilateral filter to reduce noise while preserving edges
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)

        # Adaptive thresholding to remove micro-debris and background noise
        thresh = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )

        # Morphological operations to clean small noise/debris
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

        # Convert back to BGR for PaddleOCR
        enhanced = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)

        # CLAHE — boosts local contrast for faint/micro text without blowing highlights
        lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_eq = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
        enhanced = cv2.cvtColor(cv2.merge([l_eq, a, b]), cv2.COLOR_LAB2BGR)

        # Unsharp mask — crispens micro-fonts
        blur = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=1.5)
        return cv2.addWeighted(enhanced, 1.6, blur, -0.6, 0)

    def extract_text(self, image, preprocess=True):
        """
        Returns detected text as a newline-joined string.
        Args:
            image:      BGR numpy array from OpenCV/Camera
            preprocess: apply CLAHE + unsharp preprocessing
        """
        if self.ocr_engine is None:
            logging.error("OCR engine not loaded.")
            return ""
        if image is None:
            logging.warning("Image is None.")
            return ""

        process_img = self.preprocess_image(image) if preprocess else image

        # Free any stale allocations before inference to keep peak RAM low
        gc.collect()

        logging.info("Running OCR...")
        try:
            results = self.ocr_engine.predict(process_img)

            extracted_text = []
            if results:
                for res in results:
                    texts  = res.get('rec_texts',  []) if isinstance(res, dict) else getattr(res, 'rec_texts',  [])
                    scores = res.get('rec_scores', []) if isinstance(res, dict) else getattr(res, 'rec_scores', [])
                    for text, score in zip(texts, scores):
                        try:
                            score_float = float(score)
                            # Filter: only keep words with confidence >= min_confidence
                            if score_float >= self.min_confidence:
                                logging.info(f"  '{text}' ({score_float:.2f}) ✓")
                                extracted_text.append(str(text))
                            else:
                                logging.info(f"  '{text}' ({score_float:.2f}) ✗ filtered (low confidence)")
                        except (TypeError, ValueError):
                            pass

            final_text = "\n".join(extracted_text)

            if not final_text:
                logging.info("No text detected.")
            else:
                print("\n--- OCR RESULT ---")
                print(final_text)
                print("------------------\n")

            return final_text

        except Exception as e:
            logging.error(f"OCR error: {e}")
            return ""
