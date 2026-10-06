"""Input loading shared by scripts and the pipeline."""
from pathlib import Path

import cv2
import numpy as np


def load_input(path, password=None):
    """Return (BGR image, is_digital).

    Images are read with OpenCV (is_digital=False). PDFs: page 1 is rendered at
    200 dpi (is_digital=True). e-Aadhaar PDFs are password protected, so pass
    the password if needed. Never hard-code a real password in the repo.
    """
    p = Path(path)
    if p.suffix.lower() == ".pdf":
        import fitz  # pymupdf
        with fitz.open(p) as doc:
            if doc.needs_pass and not doc.authenticate(password or ""):
                raise ValueError("PDF is password protected; supply the password")
            pix = doc[0].get_pixmap(dpi=200, alpha=False)
            img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img, True
    img = cv2.imread(str(p))
    if img is None:
        raise ValueError(f"Could not read image: {p}")
    return img, False