"""Reading data/raw/test_labels.csv and splitting it by source card.

Rules (from the project plan):
- label: genuine | forged | not_aadhaar.  Only edited CONTENT makes a card forged.
- Every image from one physical card shares a source_id, and all of them must
  land in the same subset (tune / final), or scores are inflated.
"""
import csv
import random


def load_test_labels(path):
    """Return a list of dicts with cleaned values.

    `variant_raw` keeps what was written. `variant` is corrected: a row that
    names a tamper_type but says "degraded" is treated as "edited".
    """
    rows = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            r = {k.strip(): (v or "").strip() for k, v in r.items() if k}
            r["label"] = r.get("label", "").lower()
            r["source_id"] = r.get("source_id", "").upper()
            r["variant_raw"] = r.get("variant", "").lower()
            r["variant"] = ("edited" if r.get("tamper_type") and r["variant_raw"] == "degraded"
                            else r["variant_raw"])
            rows.append(r)
    return rows


def assign_subsets(rows, tune_fraction=0.3, seed=42):
    """Return {source_id: 'tune'|'final'}.

    Whole source cards are assigned together. Sources that include a genuine
    image and sources that are forged-only are split separately, so both
    subsets contain genuine and forged cards. A subset value already written
    in the CSV is respected.
    """
    fixed = {}
    for r in rows:
        if r.get("subset") in ("tune", "final"):
            fixed[r["source_id"]] = r["subset"]

    has_genuine = {}
    for r in rows:
        has_genuine[r["source_id"]] = has_genuine.get(r["source_id"], False) or r["label"] == "genuine"

    out = dict(fixed)
    rng = random.Random(seed)
    for want_genuine in (True, False):
        ids = sorted(s for s, g in has_genuine.items() if g == want_genuine and s not in fixed)
        rng.shuffle(ids)
        n_tune = round(len(ids) * tune_fraction)
        if len(ids) >= 2:
            n_tune = max(1, n_tune)
        for i, s in enumerate(ids):
            out[s] = "tune" if i < n_tune else "final"
    return out