"""Check that test_labels.csv and the image folders agree, before evaluating.

Usage (from repo root):  python scripts/check_dataset.py
Prints ERRORS (fix these) and WARNINGS (worth a look). Exit code 1 if any error.
Never prints anything except file names, labels and ids.
"""
import argparse
import glob
import os
import sys
from collections import defaultdict

import cv2
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.common.labels import load_test_labels  # noqa: E402

LABELS = {"genuine", "forged", "not_aadhaar"}
VARIANTS = {"original", "degraded", "edited", "morphed"}
REQUIRED_COLS = ["file", "label", "source_id", "variant"]
IMG_EXT = (".jpg", ".jpeg", ".png")


def dhash(img, n=8):
    g = cv2.cvtColor(cv2.resize(img, (n + 1, n), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    return [int(x) for x in (g[:, 1:] > g[:, :-1]).flatten()]


def ham(a, b):
    return sum(x != y for x, y in zip(a, b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    a = ap.parse_args()
    with open(a.config) as f:
        cfg = yaml.safe_load(f)
    ev, root = cfg["evaluation"], cfg["data_root"]
    labels_path = os.path.join(root, ev["labels"])
    test_dir, ref_dir = os.path.join(root, ev["test_dir"]), os.path.join(root, ev["reference_dir"])

    errors, warnings = [], []
    if not os.path.exists(labels_path):
        sys.exit(f"ERROR: labels file not found: {labels_path}")
    rows = load_test_labels(labels_path)
    if not rows:
        sys.exit("ERROR: labels file has no rows")
    missing_cols = [c for c in REQUIRED_COLS if c not in rows[0]]
    if missing_cols:
        sys.exit(f"ERROR: missing columns in CSV header: {missing_cols}")

    disk = {os.path.basename(p) for p in glob.glob(os.path.join(test_dir, "*")) if p.lower().endswith(IMG_EXT)}
    seen = set()
    by_source = defaultdict(list)
    for i, r in enumerate(rows, start=2):
        f = r["file"]
        if f in seen:
            errors.append(f"line {i}: file listed twice: {f}")
        seen.add(f)
        if f not in disk:
            lower = {d.lower(): d for d in disk}
            hint = f" (found '{lower[f.lower()]}': check upper/lower case)" if f.lower() in lower else ""
            errors.append(f"line {i}: {f} is in the CSV but not in {test_dir}{hint}")
        if r["label"] not in LABELS:
            errors.append(f"line {i}: {f} has label '{r['label']}', expected one of {sorted(LABELS)}")
        if r["variant_raw"] not in VARIANTS:
            errors.append(f"line {i}: {f} has variant '{r['variant_raw']}', expected one of {sorted(VARIANTS)}")
        if not r["source_id"]:
            errors.append(f"line {i}: {f} has no source_id")
        if r["label"] == "genuine" and r.get("tamper_type"):
            warnings.append(f"{f}: genuine but has tamper_type '{r['tamper_type']}'")
        if r["variant"] != r["variant_raw"]:
            warnings.append(f"{f}: variant 'degraded' with tamper_type '{r['tamper_type']}' will be treated as 'edited'")
        by_source[r["source_id"]].append(r)

    for f in sorted(disk - seen):
        errors.append(f"{f} is in {test_dir} but has no row in the CSV")

    for s, rs in by_source.items():
        types = {r.get("card_type") for r in rs if r.get("card_type")}
        if len(types) > 1:
            warnings.append(f"{s}: same source_id but different card types {sorted(types)} "
                            f"(files {[r['file'] for r in rs]})")

    ref_paths = [p for p in sorted(glob.glob(os.path.join(ref_dir, "*"))) if p.lower().endswith(IMG_EXT)]
    if not ref_paths:
        errors.append(f"no reference images found in {ref_dir}")

    # unreadable images + near-duplicate check between reference and test
    ref_h = {}
    for p in ref_paths:
        im = cv2.imread(p)
        if im is None:
            errors.append(f"cannot read reference image {os.path.basename(p)}")
        else:
            ref_h[os.path.basename(p)] = dhash(im)
    for f in sorted(disk):
        im = cv2.imread(os.path.join(test_dir, f))
        if im is None:
            errors.append(f"cannot read test image {f}")
            continue
        h = dhash(im)
        for rn, rh in ref_h.items():
            if ham(h, rh) <= 4:
                errors.append(f"test/{f} looks like the SAME PICTURE as reference/{rn}: remove it from test")

    n_gen = sum(r["label"] == "genuine" for r in rows)
    print(f"CSV rows: {len(rows)}  genuine: {n_gen}  other: {len(rows) - n_gen}  "
          f"source cards: {len(by_source)}  genuine cards: "
          f"{len({r['source_id'] for r in rows if r['label'] == 'genuine'})}")
    print(f"Test images on disk: {len(disk)}   Reference images: {len(ref_paths)}")
    print(f"\n{len(errors)} ERROR(S)")
    for e in errors:
        print("  ERROR:", e)
    print(f"\n{len(warnings)} WARNING(S)")
    for w in warnings:
        print("  warn:", w)
    print("\nReminder: no card may appear in both folders, not even as a different photo of the same card.")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()