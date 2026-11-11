from src.common.labels import assign_subsets, load_test_labels
from src.common.metrics import choose_threshold, confusion

CSV = """file,label,source_id,variant,tamper_type,card_type,side,capture,subset,notes,review
1.jpg,genuine,s01,original,,pvc,front,phone_photo,,,
2.jpg,forged,S01,edited,text,pvc,front,phone_photo,,,
3.jpg,forged,S02,degraded,text,pvc,front,phone_photo,,,
4.jpg,forged,S03,degraded,,pvc,front,phone_photo,,,
5.jpg,genuine,S04,original,,pvc,front,phone_photo,,,
6.jpg,forged,S05,edited,text,pvc,front,phone_photo,,,
"""


def test_loader_cleans_and_fixes_variant(tmp_path):
    p = tmp_path / "l.csv"
    p.write_text(CSV)
    rows = {r["file"]: r for r in load_test_labels(p)}
    assert rows["1.jpg"]["source_id"] == "S01"            # upper-cased
    assert rows["3.jpg"]["variant"] == "edited"           # degraded + tamper_type -> edited
    assert rows["3.jpg"]["variant_raw"] == "degraded"
    assert rows["4.jpg"]["variant"] == "degraded"


def test_subsets_keep_each_source_together(tmp_path):
    p = tmp_path / "l.csv"
    p.write_text(CSV)
    rows = load_test_labels(p)
    sub = assign_subsets(rows, 0.5, 1)
    assert set(sub) == {"S01", "S02", "S03", "S04", "S05"}
    assert set(sub.values()) <= {"tune", "final"}
    assert "tune" in sub.values() and "final" in sub.values()


def test_confusion_counts():
    items = [("genuine", 0.9), ("genuine", 0.4), ("forged", 0.2), ("forged", 0.8)]
    m = confusion(items, 0.5)
    assert (m["tp"], m["fn"], m["fp"], m["tn"]) == (1, 1, 1, 1)
    assert m["frr"] == 0.5 and m["recall"] == 0.5


def test_choose_threshold_respects_frr():
    items = [("genuine", 0.9), ("genuine", 0.8), ("forged", 0.1), ("forged", 0.85)]
    t = choose_threshold(items, max_frr=0.0, step=0.05)
    m = confusion(items, t)
    assert m["frr"] == 0.0 and m["recall"] >= 0.5