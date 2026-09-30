"""Format/colour check: compare a card image against reference cards.

Tolerant by design: only hue/saturation are compared (brightness ignored),
and layout uses coarse edge-density blocks, so phone photos still pass.
"""
import cv2
import numpy as np


def _prep(img, size):
    return cv2.resize(img, tuple(size), interpolation=cv2.INTER_AREA)


def colour_similarity(img, ref, size=(856, 540)):
    """Hue+saturation histogram correlation mapped to 0..1."""
    h = []
    for im in (_prep(img, size), _prep(ref, size)):
        hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [18, 8], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 1, 0, cv2.NORM_L1)
        h.append(hist)
    corr = cv2.compareHist(h[0], h[1], cv2.HISTCMP_CORREL)
    return float(np.clip(corr, 0.0, 1.0))


def _edge_grid(img, size, grid):
    g = cv2.cvtColor(_prep(img, size), cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g, (5, 5), 0)
    e = cv2.Canny(g, 60, 160).astype(np.float32)
    cols, rows = grid
    return cv2.resize(e, (cols, rows), interpolation=cv2.INTER_AREA).flatten()


def layout_similarity(img, ref, size=(856, 540), grid=(8, 5)):
    """Correlation of coarse edge-density maps (where photo/logo/QR sit)."""
    a, b = _edge_grid(img, size, grid), _edge_grid(ref, size, grid)
    if a.std() < 1e-6 or b.std() < 1e-6:
        return 0.0
    return float(np.clip(np.corrcoef(a, b)[0, 1], 0.0, 1.0))


def _soft(x, low, high):
    return float(np.clip((x - low) / (high - low), 0.0, 1.0))


def compare_to_references(img, refs, cfg):
    """Best score across references. Returns (score, reason)."""
    if not refs:
        return 1.0, "No reference cards supplied; layout/colour check skipped"
    size, grid = tuple(cfg["card_size"]), tuple(cfg["layout_grid"])
    best = (-1.0, 0.0, 0.0)
    for ref in refs:
        c = colour_similarity(img, ref, size)
        l = layout_similarity(img, ref, size, grid)
        s = cfg["weight_colour"] * _soft(c, cfg["low"], cfg["high"]) + \
            cfg["weight_layout"] * _soft(l, cfg["low"], cfg["high"])
        if s > best[0]:
            best = (s, c, l)
    s, c, l = best
    return round(s, 3), f"Best match: colour sim {c:.2f}, layout sim {l:.2f}"
