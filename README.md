# Aadhaar Legitimacy Detection

Runs a card image through 8 checks (Phase 0-7). Each returns `(score 0-1, reason)`; Phase 7 combines them.

## Privacy rules
- NEVER commit real Aadhaar images or numbers. Use made-up numbers in tests.
- Real (masked, consented) cards live only in a restricted Google Drive folder.

## Setup (local)
    python -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    pytest

## Status
- [x] Phase 0: checksum + format/colour
- [ ] Phases 1-7
