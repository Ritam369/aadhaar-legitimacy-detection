"""Plain-Python metrics for a screening check.

Positive class = forged. Items are (label, score) pairs. A card is REJECTED
when score < threshold. Abstentions (score None) must be removed before calling.
False rejection rate (FRR) = genuine cards wrongly rejected: the costly error.
"""


def _div(a, b):
    return a / b if b else None


def confusion(items, threshold):
    tp = fn = fp = tn = 0
    for label, score in items:
        forged = label != "genuine"
        rejected = score < threshold
        if forged and rejected:
            tp += 1
        elif forged:
            fn += 1
        elif rejected:
            fp += 1
        else:
            tn += 1
    precision, recall = _div(tp, tp + fp), _div(tp, tp + fn)
    f1 = None if precision is None or recall is None or (precision + recall) == 0 \
        else 2 * precision * recall / (precision + recall)
    return {"threshold": threshold, "n": tp + fn + fp + tn, "tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "precision": precision, "recall": recall, "f1": f1,
            "frr": _div(fp, fp + tn), "accuracy": _div(tp + tn, tp + fn + fp + tn)}


def thresholds(step=0.05):
    return [round(i * step, 4) for i in range(int(round(1 / step)) + 1)]


def sweep(items, step=0.05):
    return [confusion(items, t) for t in thresholds(step)]


def choose_threshold(items, max_frr=0.2, step=0.05):
    """Highest catch rate (recall) with FRR <= max_frr; ties go to the lower threshold.
    Returns None if there are no genuine cards to measure FRR on."""
    best = None
    for m in sweep(items, step):
        if m["frr"] is None or m["frr"] > max_frr:
            continue
        key = (m["recall"] or 0.0, -m["threshold"])
        if best is None or key > best[0]:
            best = (key, m["threshold"])
    return None if best is None else best[1]