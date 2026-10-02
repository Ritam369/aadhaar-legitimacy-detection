"""Try Phase 0 on real images.

Usage (from repo root):
  python scripts/try_phase0.py --refs data/raw/reference --test data/raw/test
  python scripts/try_phase0.py --refs ... --test ... --debug-dir data/processed/debug
  python scripts/try_phase0.py --refs ... --test ... --number 234567890124
  python scripts/try_phase0.py --refs ... --test ... --corners data/raw/corners.json

--corners: JSON from scripts/corner_picker.html (hand-clicked card corners).
Images listed there are straightened from those corners; others use auto-detection.
Keys look like "reference/1.png" / "test/1.png" (folder name + file name).

--debug-dir saves each straightened card so you can SEE what the detector did.
Keep test images DIFFERENT from reference images, or scores are meaningless.
"""
import argparse
import glob
import json
import os
import sys

import cv2
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.common.preprocess import normalise_card  # noqa: E402
from src.phase0_checksum import phase0  # noqa: E402


def load_cards(folder, corners, size):
    """Return [(name, straightened_card, note)] for every image in folder."""
    tag = os.path.basename(os.path.normpath(folder))
    out = []
    for p in sorted(glob.glob(os.path.join(folder, "*"))):
        if not p.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        img = cv2.imread(p)
        if img is None:
            continue
        name = os.path.basename(p)
        crop = normalise_card(img, size, corners.get(f"{tag}/{name}"))
        out.append((name, crop.image, crop.detected, crop.note))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refs", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--number", default=None, help="made-up number to check too")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--debug-dir", default=None)
    ap.add_argument("--corners", default=None, help="corners.json from corner_picker.html")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config))["phase0"]
    cfg = dict(cfg, normalise=False)          # we straighten here ourselves
    size = tuple(cfg["card_size"])
    corners = json.load(open(a.corners)) if a.corners else {}

    refs = [c[1] for c in load_cards(a.refs, corners, size)]
    if not refs:
        sys.exit(f"No .jpg/.jpeg/.png images found in {a.refs}")
    print(f"Loaded {len(refs)} reference card(s)\n")

    tests = load_cards(a.test, corners, size)
    if not tests:
        sys.exit(f"No test images found in {a.test}")
    if a.debug_dir:
        os.makedirs(a.debug_dir, exist_ok=True)

    print(f"{'image':16} {'score':>6}  details")
    for name, card, detected, note in tests:
        res = phase0.run(a.number, card, refs, cfg)
        flag = "straightened" if detected else f"NOT detected: {note}"
        print(f"{name[:16]:16} {res.score:6.2f}  {res.reason} [{flag}; {note}]" if detected
              else f"{name[:16]:16} {res.score:6.2f}  {res.reason} [{flag}]")
        if a.debug_dir:
            cv2.imwrite(os.path.join(a.debug_dir, f"{os.path.splitext(name)[0]}_card.png"), card)


if __name__ == "__main__":
    main()
