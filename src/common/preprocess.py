"""Card detection + perspective correction.

normalise_card() finds the card outline in a photo, straightens it and returns
it at a standard size, so later phases compare like with like.
Limitations: assumes the card contrasts with its background, and does not
fix a card that is upside down (180 degrees).
"""
from dataclasses import dataclass

import cv2
import numpy as np

CARD_SIZE = (856, 540)          # (w, h), ~85.6 x 54 mm
_MIN_AREA_FRAC = 0.20           # card must cover at least 20% of the photo
_ALREADY_CROPPED_FRAC = 0.90    # above this, the photo is already the card
_ASPECT_RANGE = (1.25, 2.0)     # real card is ~1.585


@dataclass
class CardCrop:
    image: np.ndarray
    detected: bool   # True if an outline was found and straightened
    note: str


def _order(pts):
    """Order 4 points as top-left, top-right, bottom-right, bottom-left."""
    pts = np.asarray(pts, np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()   # y - x
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)],
                     pts[np.argmax(s)], pts[np.argmax(d)]], np.float32)


def _find_quad(img):
    """Return (quad in original pixel coords or None, note)."""
    h, w = img.shape[:2]
    scale = 800.0 / max(h, w) if max(h, w) > 800 else 1.0
    small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) \
        if scale != 1.0 else img
    pad = 10
    small = cv2.copyMakeBorder(small, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    sh, sw = small.shape[:2]

    gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    edges = cv2.Canny(gray, 40, 120)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, "no contours found"

    img_area = float(sh * sw)
    for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:5]:
        hull = cv2.convexHull(c)
        frac = cv2.contourArea(hull) / img_area
        if frac < _MIN_AREA_FRAC:
            break
        if frac > _ALREADY_CROPPED_FRAC:
            return None, "photo already fills the frame"
        approx = cv2.approxPolyDP(hull, 0.02 * cv2.arcLength(hull, True), True)
        quad = approx.reshape(-1, 2) if len(approx) == 4 else \
            cv2.boxPoints(cv2.minAreaRect(hull))
        quad = (np.asarray(quad, np.float32) - pad) / scale
        return quad, f"outline found ({frac:.0%} of photo)"
    return None, "no card-sized outline found"


def normalise_card(img, size=CARD_SIZE):
    """Detect, straighten and resize the card. Falls back to a plain resize."""
    quad, note = _find_quad(img)
    if quad is None:
        return CardCrop(cv2.resize(img, size, interpolation=cv2.INTER_AREA), False, note)

    tl, tr, br, bl = _order(quad)
    top, left = np.linalg.norm(tr - tl), np.linalg.norm(bl - tl)
    if left > top:                       # card photographed in portrait
        tl, tr, br, bl = tr, br, bl, tl
        top, left = left, top
    ratio = top / max(left, 1e-6)
    if not _ASPECT_RANGE[0] <= ratio <= _ASPECT_RANGE[1]:
        return CardCrop(cv2.resize(img, size, interpolation=cv2.INTER_AREA), False,
                        f"outline has odd shape (ratio {ratio:.2f}); plain resize used")

    w, h = size
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32)
    m = cv2.getPerspectiveTransform(np.array([tl, tr, br, bl], np.float32), dst)
    return CardCrop(cv2.warpPerspective(img, m, size), True, note)
