"""Phase 0 entry point: number check + optional layout/colour check.

Scores:
  invalid number            -> 0.0 (hard fail)
  valid number              -> counts as 1.0 in the average
  masked number / VID       -> left out of the average (cannot be checked)
  layout/colour             -> similarity to the best reference card, 0..1
  card not reliably located -> layout/colour skipped (never scored as fake)
Nothing to score            -> neutral 0.5

result.details holds the parts for Phase 7 / evaluation:
  checksum: valid | invalid | masked | None
  layout:   float or None
  skipped:  True if the layout/colour check was skipped
  detected: True/False if the card outline was found (None if not run)
"""
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


def run(number, image=None, refs=None, cfg=None, corners=None, refs_normalised=False):
    """number: str or None. image: BGR ndarray or None. Returns PhaseResult.

    corners: optional 4 hand-picked corners of `image` (original pixels).
    refs_normalised: True if `refs` were already straightened (saves time when
    scoring many images against the same references).
    """
    reasons, scores = [], []
    details = {"checksum": None, "layout": None, "skipped": False, "detected": None}

    if number is not None:
        ok, why = validate_aadhaar_number(number)
        if ok is False:
            details["checksum"] = "invalid"
            return PhaseResult(0.0, why, details)      # hard fail: strong red flag
        reasons.append(why)
        if ok is True:
            details["checksum"] = "valid"
            scores.append(1.0)
        else:                                          # masked / VID: not checkable
            details["checksum"] = "masked"

    if image is not None:
        if cfg is None:
            raise ValueError("cfg (config['phase0']) required when image is given")
        size = tuple(cfg["card_size"])
        note, reliable = "", True
        if cfg.get("normalise", True):
            crop = normalise_card(image, size, corners)
            image, reliable = crop.image, crop.reliable
            details["detected"] = crop.detected
            note = f" [card {'straightened' if crop.detected else 'not detected: ' + crop.note}]"
            if not refs_normalised:
                refs = [normalise_card(r, size).image for r in (refs or [])]
        if reliable:
            # The straightener cannot tell which way up the card is, so also try
            # the image rotated 180 degrees and keep the better match.
            s, why = max(
                (compare_to_references(im, refs or [], cfg)
                 for im in (image, cv2.rotate(image, cv2.ROTATE_180))),
                key=lambda t: t[0])
            scores.append(s)
            details["layout"] = s
            reasons.append(why + note)
        else:
            # Not card-shaped and no outline found: a layout score here would be
            # noise and could wrongly reject a genuine user. Skip it.
            details["skipped"] = True
            reasons.append("Layout/colour check skipped, card not reliably located" + note)

    if not scores:
        return PhaseResult.neutral("; ".join(reasons) or "No number or image supplied", details)
    return PhaseResult(round(sum(scores) / len(scores), 3), "; ".join(reasons), details)