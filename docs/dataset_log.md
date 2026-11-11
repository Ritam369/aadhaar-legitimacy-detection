## Phase 0 dataset (add to the end of docs/dataset_log.md)

- Reference folder `data/raw/reference/`: genuine cards only (no CSV; every file counts as genuine).
- Test folder `data/raw/test/`: genuine and forged images mixed, labelled in `data/raw/test_labels.csv`.
- Columns: file, label (genuine|forged|not_aadhaar), source_id, variant (original|degraded|edited),
  tamper_type, card_type, side, capture, subset (tune|final, assigned by script), notes.
- source_id = one physical card; every copy/edit/photo of it shares the id and the same subset.
- Rule: only edited CONTENT makes a card forged; blur/darkening/xerox/scan of an unchanged card is genuine.
- Known issue: many rows are labelled forged with variant "degraded"; some may be genuine cards in poor quality.
  Rows naming a tamper_type with variant "degraded" are treated as "edited" by the code.
- Consent: cards used only from people who agreed; first 8 digits masked where possible; stored on restricted Drive, never on GitHub.
- Counts and dates: (fill in)