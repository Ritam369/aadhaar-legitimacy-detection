"""Reading data/raw/test_labels.csv and splitting it by source card.

Rules (from the project plan):
- label: genuine | forged | not_aadhaar | unknown.  Only edited CONTENT makes a card forged.
  'unknown' = we do not know whether it is genuine or forged: such rows are kept in the
  file but EXCLUDED from every metric.
- Every image from one physical card shares a source_id, and all of them must
  land in the same subset (tune / final), or scores are inflated.
- Free-text values ("paper-cut soft copy", "front & back", "phone photo") are normalised
  to simple tokens (paper_cut_softcopy, both, phone_photo), so the CSV can be typed naturally.
"""
import csv
import random
import re

KNOWN_LABELS = ("genuine", "forged", "not_aadhaar")
_VARIANT_MAP = {"blurred": "degraded", "blur": "degraded", "noisy": "degraded", "dark": "degraded"}


def _tok(s):
    s = (s or "").strip().lower().replace("&", " and ")
    s = re.sub(r"[\s\-/]+", "_", s)
    return re.sub(r"_+", "_", s).strip("_")


def _card_type(s):
    t = _tok(s)
    for a, b in (("soft_copy", "softcopy"), ("xerox_copy", "xerox"), ("letter_format", "letter")):
        t = t.replace(a, b)
    return t


def _side(s):
    t = _tok(s)
    return "both" if t in ("front_and_back", "front_back", "back_and_front", "both") else t


def load_test_labels(path):
    """Return a list of dicts with cleaned values.

    - label / source_id / card_type / side / capture / tamper_type are normalised.
    - `capture` is read from either a `capture` or a `capture_type` column.
    - `variant_written` is what was typed; `variant_raw` maps blurred -> degraded;
      `variant` is corrected: a row that names a tamper_type but says "degraded"
      is treated as "edited".
    """
    rows = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            r = {k.strip(): (v or "").strip() for k, v in r.items() if k}
            r["file"] = r.get("file", "")
            r["label"] = _tok(r.get("label", ""))
            r["source_id"] = r.get("source_id", "").upper()
            r["variant_written"] = _tok(r.get("variant", ""))
            r["variant_raw"] = _VARIANT_MAP.get(r["variant_written"], r["variant_written"])
            r["tamper_type"] = _tok(r.get("tamper_type", ""))
            r["variant"] = ("edited" if r["tamper_type"] and r["variant_raw"] == "degraded"
                            else r["variant_raw"])
            r["card_type"] = _card_type(r.get("card_type", ""))
            r["side"] = _side(r.get("side", ""))
            r["capture"] = _tok(r.get("capture") or r.get("capture_type") or "")
            rows.append(r)
    return rows


def assign_subsets(rows, tune_fraction=0.3, seed=42):
    """Return {source_id: 'tune'|'final'|'excluded'}.

    Whole source cards are assigned together. Sources that include a genuine
    image and sources that are forged-only are split separately, so both
    subsets contain genuine and forged cards. Sources that have only 'unknown'
    rows are 'excluded'. A subset value already written in the CSV is respected.
    """
    known = [r for r in rows if r["label"] in KNOWN_LABELS]
    fixed = {}
    for r in known:
        if r.get("subset") in ("tune", "final"):
            fixed[r["source_id"]] = r["subset"]

    has_genuine = {}
    for r in known:
        has_genuine[r["source_id"]] = has_genuine.get(r["source_id"], False) or r["label"] == "genuine"

    out = {r["source_id"]: "excluded" for r in rows}
    out.update(fixed)
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