"""Try Phase 0 on real images.

Usage (from repo root):
  python scripts/try_phase0.py --refs data/raw/reference --test data/raw/test
  python scripts/try_phase0.py --refs data/raw/reference --test data/raw/test --number 234567890124

Every image in --test is compared against every image in --refs.
Keep test images DIFFERENT from reference images, or the score is meaningless.
"""
import argparse
import glob
import os
import sys

import cv2
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.phase0_checksum import phase0  # noqa: E402
from src.phase0_checksum.layout_colour import colour_similarity, layout_similarity  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refs", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--number", default=None, help="made-up/own number to check too")
    ap.add_argument("--config", default="config.yaml")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config))["phase0"]
    refs = phase0.load_references(a.refs)
    if not refs:
        sys.exit(f"No .jpg/.jpeg/.png images found in {a.refs}")
    print(f"Loaded {len(refs)} reference card(s)\n")

    size, grid = tuple(cfg["card_size"]), tuple(cfg["layout_grid"])
    paths = sorted(p for p in glob.glob(os.path.join(a.test, "*"))
                   if p.lower().endswith((".jpg", ".jpeg", ".png")))
    if not paths:
        sys.exit(f"No test images found in {a.test}")

    print(f"{'image':30} {'colour':>7} {'layout':>7} {'score':>6}  reason")
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            print(f"{os.path.basename(p):30} could not be read")
            continue
        c = max(colour_similarity(img, r, size) for r in refs)
        l = max(layout_similarity(img, r, size, grid) for r in refs)
        res = phase0.run(a.number, img, refs, cfg)
        print(f"{os.path.basename(p)[:30]:30} {c:7.2f} {l:7.2f} {res.score:6.2f}  {res.reason}")


if __name__ == "__main__":
    main()
