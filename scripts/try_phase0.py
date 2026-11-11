"""Try Phase 0 on real images (quick look, no labels needed).

Usage (from repo root):
  python scripts/try_phase0.py --refs data/raw/reference --test data/raw/test
  python scripts/try_phase0.py --refs ... --test ... --debug-dir data/processed/debug
  python scripts/try_phase0.py --refs ... --test ... --number "XXXX XXXX 0124"
  python scripts/try_phase0.py --refs ... --test ... --corners data/raw/corners.json

--corners: JSON from scripts/corner_picker.html (hand-clicked card corners).
Keys look like "reference/1.png" / "test/1.png" (folder name + file name).
--debug-dir saves each straightened card so you can SEE what the detector did.
A line saying SKIPPED means the card could not be located and no layout score was given.
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


def list_images(folder):
    return [p for p in sorted(glob.glob(os.path.join(folder, "*")))
            if p.lower().endswith((".jpg", ".jpeg", ".png"))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refs", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--number", default=None, help="made-up number to check too")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--debug-dir", default=None)
    ap.add_argument("--corners", default=None, help="corners.json from corner_picker.html")
    a = ap.parse_args()

    with open(a.config) as f:
        cfg = yaml.safe_load(f)["phase0"]
    size = tuple(cfg["card_size"])
    corners = json.load(open(a.corners)) if a.corners else {}

    ref_tag = os.path.basename(os.path.normpath(a.refs))
    refs = []
    for p in list_images(a.refs):
        img = cv2.imread(p)
        if img is not None:
            refs.append(normalise_card(img, size, corners.get(f"{ref_tag}/{os.path.basename(p)}")).image)
    if not refs:
        sys.exit(f"No .jpg/.jpeg/.png images found in {a.refs}")
    print(f"Loaded {len(refs)} reference card(s)\n")

    tag = os.path.basename(os.path.normpath(a.test))
    paths = list_images(a.test)
    if not paths:
        sys.exit(f"No test images found in {a.test}")
    if a.debug_dir:
        os.makedirs(a.debug_dir, exist_ok=True)

    print(f"{'image':16} {'score':>6}  details")
    for p in paths:
        name = os.path.basename(p)
        img = cv2.imread(p)
        if img is None:
            print(f"{name[:16]:16} {'-':>6}  could not read image")
            continue
        c = corners.get(f"{tag}/{name}")
        res = phase0.run(a.number, img, refs, cfg, corners=c, refs_normalised=True)
        flag = "SKIPPED " if res.details.get("skipped") else ""
        print(f"{name[:16]:16} {res.score:6.2f}  {flag}{res.reason}")
        if a.debug_dir:
            crop = normalise_card(img, size, c)
            cv2.imwrite(os.path.join(a.debug_dir, f"{os.path.splitext(name)[0]}_card.png"), crop.image)


if __name__ == "__main__":
    main()