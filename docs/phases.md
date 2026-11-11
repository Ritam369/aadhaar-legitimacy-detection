# Phases
See Aadhaar_Project_Plan.md. Every phase returns PhaseResult(score, reason) from src/common/result.py.

## Phase 0: Aadhaar number checksum + format/colour check

**Code:** `src/phase0_checksum/` (verhoeff.py, layout_colour.py, phase0.py), card straightening in `src/common/preprocess.py`.

**What it checks**
1. Number: 12 digits, first digit not 0/1, Verhoeff checksum.
2. Layout and colour of the card against the reference cards in `data/raw/reference/`
   (hue/saturation histogram + coarse edge-density grid; brightness ignored; tolerant of phone photos).

**Scoring**
| Case | Result |
|---|---|
| Invalid number | 0.0 (hard fail) |
| Valid number | counts as 1.0 in the average |
| Masked number (XXXX XXXX 1234) or 16-digit Virtual ID | not checkable: left out of the average, never a failure |
| Card could not be located and image is not card-shaped | layout/colour skipped (abstain), never scored as fake |
| Nothing to score | neutral 0.5 |

`PhaseResult.details` holds `checksum`, `layout`, `skipped`, `detected` for Phase 7.

**Limits (say these in the report)**
- A valid checksum only means the number is well-formed, not that it was issued.
- A masked card cannot have its checksum checked; the last 4 digits are cross-checked in Phase 1 (QR).
- Layout/colour is a weak signal. Edited text or photos usually pass Phase 0; Phases 3 to 6 handle those.
- Evaluated on a small, partly synthetic set with no real-world forgeries.

**How to run**
```
python scripts/try_phase0.py --refs data/raw/reference --test data/raw/test
python scripts/check_dataset.py
python scripts/evaluate_phase0.py
```