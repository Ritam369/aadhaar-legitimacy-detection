"""Phase 0 entry point: number check + optional layout/colour check."""
import glob
import os

import cv2

from src.common.result import PhaseResult
from src.phase0_checksum.layout_colour import compare_to_references
from src.phase0_checksum.verhoeff import validate_aadhaar_number


def load_references(folder):
    refs = []
    for p in sorted(glob.glob(os.path.join(folder, "*"))):
        if p.lower().endswith((".jpg", ".jpeg", ".png")):
            im = cv2.imread(p)
            if im is not None:
                refs.append(im)
    return refs


def run(number, image=None, refs=None, cfg=None):
    """number: str or None. image: BGR ndarray or None. Returns PhaseResult."""
    reasons, scores = [], []
    if number is not None:
        ok, why = validate_aadhaar_number(number)
        if not ok:
            return PhaseResult(0.0, why)   # hard fail: strong red flag
        reasons.append(why)
        scores.append(1.0)
    if image is not None:
        if cfg is None:
            raise ValueError("cfg (config['phase0']) required when image is given")
        s, why = compare_to_references(image, refs or [], cfg)
        scores.append(s)
        reasons.append(why)
    if not scores:
        return PhaseResult(0.0, "No number or image supplied")
    return PhaseResult(round(sum(scores) / len(scores), 3), "; ".join(reasons))
