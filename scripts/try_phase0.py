"""Try Phase 0 on real images.

Usage (from repo root):
  python scripts/try_phase0.py --refs data/raw/reference --test data/raw/test
  python scripts/try_phase0.py --refs ... --test ... --debug-dir data/processed/debug
  python scripts/try_phase0.py --refs ... --test ... --number 234567890124

--debug-dir saves each straightened card so you can SEE what the detector did.
Keep test images DIFFERENT from reference images, or scores are meaningless.
"""
import argparse
import glob
import os
import sys

import cv2
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.common.preprocess import normalise_card  # noqa: E402
from src.phase0_checksum import phase0  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refs", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--number", default=None, help="made-up number to check too")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--debug-dir", default=None)
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config))["phase0"]
    refs = phase0.load_references(a.refs)
    if not refs:
        sys.exit(f"No .jpg/.jpeg/.png images found in {a.refs}")
    print(f"Loaded {len(refs)} reference card(s)\n")

    paths = sorted(p for p in glob.glob(os.path.join(a.test, "*"))
                   if p.lower().endswith((".jpg", ".jpeg", ".png")))
    if not paths:
        sys.exit(f"No test images found in {a.test}")
    if a.debug_dir:
        os.makedirs(a.debug_dir, exist_ok=True)

    print(f"{'image':16} {'score':>6}  details")
    for p in paths:
        img = cv2.imread(p)
        name = os.path.basename(p)
        if img is None:
            print(f"{name[:16]:16} could not be read")
            continue
        res = phase0.run(a.number, img, refs, cfg)
        print(f"{name[:16]:16} {res.score:6.2f}  {res.reason}")
        if a.debug_dir:
            cv2.imwrite(os.path.join(a.debug_dir, f"{os.path.splitext(name)[0]}_card.png"),
                        normalise_card(img, tuple(cfg["card_size"])).image)


if __name__ == "__main__":
    main()
