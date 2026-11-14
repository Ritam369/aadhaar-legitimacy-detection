"""Score every test image with Phase 0 and report metrics.

Usage (from repo root):  python scripts/evaluate_phase0.py

What it does:
 1. Reads data/raw/test_labels.csv and splits source cards into 'tune' and 'final'
    (whole source cards stay together).
 2. Runs Phase 0 on each test image against the reference cards (no number supplied).
 3. Picks the accept/reject threshold on the 'tune' subset only: highest catch rate
    while wrongly rejecting at most evaluation.max_frr of genuine cards.
 4. Reports the numbers on 'final' (and 'tune') with that threshold.
Images where the card could not be located are ABSTAINED, not scored as fake.
Writes results/metrics/phase0_predictions.csv and phase0_metrics.csv.
Phase 0 only sees layout/colour, so it is NOT expected to catch edited text or photos.
"""
import argparse
import csv
import os
import sys
from collections import defaultdict

import cv2
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.common.labels import KNOWN_LABELS, assign_subsets, load_test_labels  # noqa: E402
from src.common.metrics import choose_threshold, confusion, sweep  # noqa: E402
from src.common.preprocess import normalise_card  # noqa: E402
from src.phase0_checksum import phase0  # noqa: E402


def f2(x):
    return "  -  " if x is None else f"{x:5.2f}"


def show(title, m):
    print(f"{title:28} n={m['n']:3}  TP={m['tp']:3} FN={m['fn']:3} FP={m['fp']:3} TN={m['tn']:3}  "
          f"precision={f2(m['precision'])} recall(catch)={f2(m['recall'])} F1={f2(m['f1'])} "
          f"FRR={f2(m['frr'])} acc={f2(m['accuracy'])}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--out", default="results/metrics")
    a = ap.parse_args()
    with open(a.config) as f:
        cfg = yaml.safe_load(f)
    p0, ev, root = cfg["phase0"], cfg["evaluation"], cfg["data_root"]
    size = tuple(p0["card_size"])

    refs = phase0.load_references(os.path.join(root, ev["reference_dir"]))
    if not refs:
        sys.exit("No reference images found.")
    refs = [normalise_card(r, size).image for r in refs]

    rows = load_test_labels(os.path.join(root, ev["labels"]))
    subsets = assign_subsets(rows, ev["tune_fraction"], ev["seed"])

    preds, missing = [], []
    for r in rows:
        img = cv2.imread(os.path.join(root, ev["test_dir"], r["file"]))
        if img is None:
            missing.append(r["file"])
            continue
        res = phase0.run(None, img, refs, p0, refs_normalised=True)
        abstain = bool(res.details.get("skipped"))
        preds.append({"file": r["file"], "label": r["label"], "variant": r["variant"],
                      "card_type": r.get("card_type", ""), "side": r.get("side", ""),
                      "capture": r.get("capture", ""), "source_id": r["source_id"],
                      "subset": subsets.get(r["source_id"], "excluded"),
                      "score": "" if abstain else res.score, "abstained": int(abstain),
                      "detected": res.details.get("detected"), "reason": res.reason})
    if missing:
        print(f"WARNING: {len(missing)} images could not be read (run check_dataset.py): {missing[:8]}\n")

    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "phase0_predictions.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(preds[0].keys()))
        w.writeheader()
        w.writerows(preds)

    def items(subset=None):
        return [(p["label"], p["score"]) for p in preds
                if p["label"] in KNOWN_LABELS and p["score"] != ""
                and (subset is None or p["subset"] == subset)]

    labelled = [p for p in preds if p["label"] in KNOWN_LABELS]
    n_unk = len(preds) - len(labelled)
    n_abs = sum(p["abstained"] for p in labelled)
    print(f"Labelled images: {len(labelled)}   unknown (left out of all metrics): {n_unk}")
    print(f"Labelled images scored: {len(labelled) - n_abs}   abstained (card not located): {n_abs} "
          f"({n_abs / max(1, len(labelled)):.0%})")
    for s in ("tune", "final"):
        sub = [p for p in labelled if p["subset"] == s]
        print(f"  {s}: {len(sub)} images, {len({p['source_id'] for p in sub})} source cards, "
              f"{sum(p['label'] == 'genuine' for p in sub)} genuine images")

    t = choose_threshold(items("tune"), ev["max_frr"], ev["threshold_step"])
    if t is None:
        t = p0["threshold"]
        print(f"\nCould not tune a threshold (no genuine cards in 'tune'). Using config value {t}.")
    print(f"\nChosen threshold (reject if score < t): {t}   "
          f"[tuned on 'tune' with FRR <= {ev['max_frr']}]\n")
    show("TUNE  (used to choose t)", confusion(items("tune"), t))
    show("FINAL (honest numbers)", confusion(items("final"), t))
    show("ALL", confusion(items(), t))

    print("\nCatch rate on forged images by kind (all subsets, threshold above):")
    groups = defaultdict(list)
    for p in preds:
        if p["label"] in ("forged", "not_aadhaar") and p["score"] != "":
            groups[("variant", p["variant"])].append(p["score"] < t)
            groups[("card_type", p["card_type"])].append(p["score"] < t)
    for (k, v), hits in sorted(groups.items()):
        print(f"  {k:10} {v:22} caught {sum(hits):3}/{len(hits):3} = {sum(hits) / len(hits):.2f}")
    gen = [p["score"] < t for p in preds if p["label"] == "genuine" and p["score"] != ""]
    if gen:
        print(f"\nGenuine images wrongly rejected: {sum(gen)}/{len(gen)}  (this is the costly error)")

    with open(os.path.join(a.out, "phase0_metrics.csv"), "w", newline="") as f:
        cols = ["subset", "threshold", "n", "tp", "fn", "fp", "tn", "precision", "recall", "f1", "frr", "accuracy"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for s in ("tune", "final", "all"):
            for m in sweep(items(None if s == "all" else s), ev["threshold_step"]):
                w.writerow({"subset": s, **{k: m[k] for k in cols[1:]}})
    print(f"\nSaved {a.out}/phase0_predictions.csv and phase0_metrics.csv")


if __name__ == "__main__":
    main()