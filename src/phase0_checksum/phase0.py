"""Phase 0 entry point: number check + optional layout/colour check."""
import glob
import os
 
import cv2
 
from src.common.preprocess import normalise_card
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
        size = tuple(cfg["card_size"])
        note, reliable = "", True
        if cfg.get("normalise", True):
            crop = normalise_card(image, size)
            image, reliable = crop.image, crop.reliable
            note = f" [card {'straightened' if crop.detected else 'not detected: ' + crop.note}]"
            refs = [normalise_card(r, size).image for r in (refs or [])]
        if reliable:
            # The straightener cannot tell which way up the card is, so also try
            # the image rotated 180 degrees and keep the better match.
            s, why = max(
                (compare_to_references(im, refs or [], cfg)
                 for im in (image, cv2.rotate(image, cv2.ROTATE_180))),
                key=lambda t: t[0])
            scores.append(s)
            reasons.append(why + note)
        else:
            # Not card-shaped and no outline found: a layout score here would be
            # noise and could wrongly reject a genuine user. Skip it.
            reasons.append("Layout/colour check skipped, card not reliably located" + note)
    if not scores:
        return PhaseResult.neutral("; ".join(reasons) or "No number or image supplied")
    return PhaseResult(round(sum(scores) / len(scores), 3), "; ".join(reasons))
