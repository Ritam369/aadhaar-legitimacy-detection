"""Runs the enabled phases on one input.

Add a phase = write a _run_phaseN(img, is_digital, number, cfg) function and
add one line to RUNNERS. Usage:  python -m src.pipeline path/to/card.png [number]
"""
import sys

import yaml

from src.common.utils import load_input
from src.phase0_checksum import phase0


def _run_phase0(img, is_digital, number, cfg):
    refs = phase0.load_references(f'{cfg["data_root"]}/raw/reference')
    # Digital PDFs are not photographed cards: skip layout/colour, keep number check.
    return phase0.run(number, None if is_digital else img, refs, cfg["phase0"])


RUNNERS = {"phase0": _run_phase0}


def run_pipeline(path, number=None, cfg_path="config.yaml"):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    img, is_digital = load_input(path)
    results = {}
    for name, fn in RUNNERS.items():
        if cfg["phases"].get(name):
            results[name] = fn(img, is_digital, number, cfg)
    return results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python -m src.pipeline <image-or-pdf> [aadhaar number]")
    for k, r in run_pipeline(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None).items():
        print(k, r.score, r.reason)