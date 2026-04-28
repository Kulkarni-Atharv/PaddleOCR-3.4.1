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
    Original version without preprocessing or filtering.
    """
    def __init__(self, lang='en'):
        self.lang = lang
        self.ocr_engine = None
        self._initialize_engine()

    def _initialize_engine(self):
        if PaddleOCR is None:
            logging.error("Cannot initialize: paddleocr is missing.")
            return

        logging.info("Initializing PaddleOCR...")
        try:
            self.ocr_engine = PaddleOCR(
                lang=self.lang,
                device='cpu',
                cpu_threads=1,
                use_angle_cls=True,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                det_limit_side_len=640,
                rec_batch_num=1,
            )
            logging.info("PaddleOCR initialized.")
        except Exception as e:
            logging.error(f"Failed to initialize PaddleOCR: {e}")

    def extract_text(self, image):
        """
        Returns detected text as a newline-joined string.
        """
        if self.ocr_engine is None:
            logging.error("OCR engine not loaded.")
            return ""
        if image is None:
            logging.warning("Image is None.")
            return ""

        # Resize image to reduce memory usage on Pi CM5 (4GB RAM)
        # Original: 1456x1088, resize to 800x600
        h, w = image.shape[:2]
        if w > 800 or h > 600:
            image = cv2.resize(image, (800, 600), interpolation=cv2.INTER_AREA)
            logging.info(f"Resized image from {w}x{h} to 800x600")

        gc.collect()

        logging.info("Running OCR...")
        try:
            results = self.ocr_engine.predict(image)

            extracted_text = []
            if results:
                for res in results:
                    texts = res.get('rec_texts', []) if isinstance(res, dict) else getattr(res, 'rec_texts', [])
                    for text in texts:
                        extracted_text.append(str(text))

            final_text = "\n".join(extracted_text)

            if not final_text:
                logging.info("No text detected.")
            else:
                print("\n--- OCR RESULT ---")
                print(final_text)
                print("------------------\n")

            return final_text
        except Exception as e:
            logging.error(f"OCR failed: {e}")
            return ""

            return final_text

        except Exception as e:
            logging.error(f"OCR error: {e}")
            return ""
