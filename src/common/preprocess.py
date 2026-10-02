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
_MIN_AREA_FRAC = 0.02           # card must cover at least 2% of the photo
_ALREADY_CROPPED_FRAC = 0.90    # above this, the photo is already the card
_MIN_RECTANGULARITY = 0.60      # contour area / its bounding rotated rectangle
_ASPECT_RANGE = (1.2, 3.3)      # real card is ~1.585; oblique photos stretch it
_CARD_ASPECT = 1.585


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


def _masks(gray):
    """Several ways of turning the photo into shapes; each may catch the card."""
    k3, k5 = np.ones((3, 3), np.uint8), np.ones((5, 5), np.uint8)
    edges = cv2.Canny(gray, 40, 120)
    for it in (1, 2):
        yield cv2.morphologyEx(cv2.dilate(edges, k3, iterations=it), cv2.MORPH_CLOSE, k5)
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    for m in (otsu, 255 - otsu):
        yield cv2.morphologyEx(m, cv2.MORPH_OPEN, k5)


def _find_quad(img):
    """Return (quad in original pixel coords or None, note).

    Collects card-like shapes (rectangular, card-ish aspect, big enough) from
    several segmentations and picks the best-scoring one.
    """
    h, w = img.shape[:2]
    scale = 800.0 / max(h, w) if max(h, w) > 800 else 1.0
    small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) \
        if scale != 1.0 else img
    pad = 10
    small = cv2.copyMakeBorder(small, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    sh, sw = small.shape[:2]
    img_area = float(sh * sw)
    gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (5, 5), 0)

    best, best_score, best_frac, saw_full = None, 0.0, 0.0, False
    for mask in _masks(gray):
        cnts, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            area = cv2.contourArea(c)
            frac = area / img_area
            if frac < _MIN_AREA_FRAC:
                continue
            if frac > _ALREADY_CROPPED_FRAC:
                saw_full = True
                continue
            (_, _), (rw, rh), _ = cv2.minAreaRect(c)
            if min(rw, rh) < 1:
                continue
            rect = area / (rw * rh)
            asp = max(rw, rh) / min(rw, rh)
            if rect < _MIN_RECTANGULARITY or not _ASPECT_RANGE[0] <= asp <= _ASPECT_RANGE[1]:
                continue
            aspect_pen = np.exp(-0.5 * abs(np.log(asp / _CARD_ASPECT)))
            score = rect * np.sqrt(frac) * aspect_pen
            if score > best_score:
                hull = cv2.convexHull(c)
                approx = cv2.approxPolyDP(hull, 0.03 * cv2.arcLength(hull, True), True)
                quad = approx.reshape(-1, 2) if len(approx) == 4 else \
                    cv2.boxPoints(cv2.minAreaRect(c))
                best, best_score, best_frac, note = quad, score, frac, \
                    f"outline found ({frac:.0%} of photo, rectangularity {rect:.2f})"
    # An image that is itself card-shaped, with only a small rectangle inside,
    # is most likely an already-cropped card (the rectangle is its photo/QR box).
    frame_asp = max(h, w) / min(h, w)
    if best is not None and saw_full and best_frac < 0.25 and 1.52 <= frame_asp <= 1.65:
        return None, "image is already card-shaped"
    if best is None:
        return None, ("photo already fills the frame" if saw_full
                      else "no card-like outline found")
    return (np.asarray(best, np.float32) - pad) / scale, note


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

    w, h = size
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32)
    m = cv2.getPerspectiveTransform(np.array([tl, tr, br, bl], np.float32), dst)
    return CardCrop(cv2.warpPerspective(img, m, size), True, note)
