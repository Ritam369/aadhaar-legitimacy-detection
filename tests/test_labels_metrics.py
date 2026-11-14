from src.common.labels import assign_subsets, load_test_labels
from src.common.metrics import choose_threshold, confusion

CSV = """file,label,source_id,variant,tamper_type,card_type,side,capture_type,subset,notes
1.jpg,genuine,s01,original,,paper-cut soft copy,front & back,phone photo,,
2.jpg,forged,S01,edited,text,pvc,front,phone photo,,
3.jpg,forged,S02,degraded,text,pvc,front,phone photo,,
4.jpg,forged,S03,blurred,,laminated paper-cut,front,phone photo,,
5.jpg,genuine,S04,original,,pvc,front,phone photo,,
6.jpg,forged,S05,edited,text,pvc,front,phone photo,,
7.jpg,unknown,S06,,,letter format xerox copy,front,phone photo,,
8.jpg,unknown,S01,,,pvc,front,phone photo,,
"""


def _rows(tmp_path):
    p = tmp_path / "l.csv"
    p.write_text(CSV)
    return {r["file"]: r for r in load_test_labels(p)}


def test_loader_normalises_values(tmp_path):
    rows = _rows(tmp_path)
    assert rows["1.jpg"]["source_id"] == "S01"                      # upper-cased
    assert rows["1.jpg"]["card_type"] == "paper_cut_softcopy"
    assert rows["1.jpg"]["side"] == "both"                          # "front & back"
    assert rows["1.jpg"]["capture"] == "phone_photo"                # from capture_type column
    assert rows["4.jpg"]["variant"] == "degraded"                   # "blurred" -> degraded
    assert rows["4.jpg"]["card_type"] == "laminated_paper_cut"


def test_loader_fixes_degraded_with_tamper(tmp_path):
    rows = _rows(tmp_path)
    assert rows["3.jpg"]["variant"] == "edited"
    assert rows["3.jpg"]["variant_raw"] == "degraded"


def test_unknown_rows_allowed_blank_variant(tmp_path):
    rows = _rows(tmp_path)
    assert rows["7.jpg"]["label"] == "unknown" and rows["7.jpg"]["variant"] == ""


def test_subsets_keep_each_source_together(tmp_path):
    rows = list(_rows(tmp_path).values())
    sub = assign_subsets(rows, 0.5, 1)
    assert sub["S06"] == "excluded"                                 # unknown-only source
    assert {sub[s] for s in ("S01", "S02", "S03", "S04", "S05")} <= {"tune", "final"}
    assert "tune" in sub.values() and "final" in sub.values()


def test_confusion_counts_and_ignores_unknown():
    items = [("genuine", 0.9), ("genuine", 0.4), ("forged", 0.2), ("forged", 0.8), ("unknown", 0.0)]
    m = confusion(items, 0.5)
    assert (m["tp"], m["fn"], m["fp"], m["tn"]) == (1, 1, 1, 1)
    assert m["frr"] == 0.5 and m["recall"] == 0.5 and m["n"] == 4


def test_choose_threshold_respects_frr():
    items = [("genuine", 0.9), ("genuine", 0.8), ("forged", 0.1), ("forged", 0.85)]
    t = choose_threshold(items, max_frr=0.0, step=0.05)
    m = confusion(items, t)
    assert m["frr"] == 0.0 and m["recall"] >= 0.5