"""
Metrics for the VLM-vs-Gemini comparison.

Deliberately stdlib-only (no sklearn/numpy) so the scoring logic can be unit
tested anywhere, including on a machine with no ML stack installed.

Two things here are worth more than the metrics themselves:

  * Every point estimate ships with a bootstrap confidence interval. At n=15
    held-out pages a bare "94% accurate" is indistinguishable from "73%
    accurate"; the interval is the honest object.
  * paired_bootstrap_delta tests the *difference* between two systems on the
    same items, which is the question actually being asked ("is the LoRA model
    worse than Gemini, and by how much"). Comparing two independent CIs is a
    weaker and commonly-botched substitute.
"""

from __future__ import annotations

import math
import random
import re
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher

DEFAULT_RESAMPLES = 10_000
DEFAULT_SEED = 20260731


# ── Classification ─────────────────────────────────────────────────────────
@dataclass
class ClassificationResult:
    n: int
    accuracy: float
    accuracy_ci95: tuple[float, float]
    macro_f1: float
    macro_f1_ci95: tuple[float, float]
    per_class: dict = field(default_factory=dict)
    confusion: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _macro_f1(pairs: list[tuple[str, str]], classes: list[str]) -> float:
    f1s = []
    for c in classes:
        tp = sum(1 for g, p in pairs if g == c and p == c)
        fp = sum(1 for g, p in pairs if g != c and p == c)
        fn = sum(1 for g, p in pairs if g == c and p != c)
        if tp + fp + fn == 0:
            continue  # class absent from both gold and pred; skip, don't score 0
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return sum(f1s) / len(f1s) if f1s else 0.0


def _accuracy(pairs: list[tuple[str, str]]) -> float:
    return sum(1 for g, p in pairs if g == p) / len(pairs) if pairs else 0.0


def bootstrap_ci(
    pairs: list,
    stat_fn,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile bootstrap CI for any statistic over a list of per-item records."""
    if len(pairs) < 2:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(pairs)
    stats = []
    for _ in range(resamples):
        sample = [pairs[rng.randrange(n)] for _ in range(n)]
        stats.append(stat_fn(sample))
    stats.sort()
    lo = stats[int((alpha / 2) * resamples)]
    hi = stats[min(resamples - 1, int((1 - alpha / 2) * resamples))]
    return (round(lo, 4), round(hi, 4))


def score_classification(
    gold: list[str],
    pred: list[str],
    classes: list[str],
    resamples: int = DEFAULT_RESAMPLES,
) -> ClassificationResult:
    if len(gold) != len(pred):
        raise ValueError(f"length mismatch: {len(gold)} gold vs {len(pred)} pred")
    pairs = list(zip(gold, pred))

    per_class = {}
    for c in classes:
        tp = sum(1 for g, p in pairs if g == c and p == c)
        fp = sum(1 for g, p in pairs if g != c and p == c)
        fn = sum(1 for g, p in pairs if g == c and p != c)
        support = sum(1 for g, _ in pairs if g == c)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        per_class[c] = {
            "support": support,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(2 * prec * rec / (prec + rec), 4) if prec + rec else 0.0,
        }

    confusion: dict[str, dict[str, int]] = {g: {} for g in classes}
    for g, p in pairs:
        confusion.setdefault(g, {})
        confusion[g][p] = confusion[g].get(p, 0) + 1

    return ClassificationResult(
        n=len(pairs),
        accuracy=round(_accuracy(pairs), 4),
        accuracy_ci95=bootstrap_ci(pairs, _accuracy, resamples),
        macro_f1=round(_macro_f1(pairs, classes), 4),
        macro_f1_ci95=bootstrap_ci(pairs, lambda s: _macro_f1(s, classes), resamples),
        per_class=per_class,
        confusion=confusion,
    )


def paired_bootstrap_delta(
    gold: list[str],
    pred_a: list[str],
    pred_b: list[str],
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> dict:
    """
    Paired bootstrap on accuracy(A) - accuracy(B) over the same items.

    Returns the point delta, its 95% CI, and a two-sided p-value for
    H0: no difference. If the CI straddles zero you have not shown a difference,
    however suggestive the point estimate looks.
    """
    records = list(zip(gold, pred_a, pred_b))
    if len(records) < 2:
        return {"delta": float("nan"), "ci95": (float("nan"), float("nan")), "p_value": float("nan")}

    def delta(sample):
        acc_a = sum(1 for g, a, _ in sample if g == a) / len(sample)
        acc_b = sum(1 for g, _, b in sample if g == b) / len(sample)
        return acc_a - acc_b

    observed = delta(records)
    rng = random.Random(seed)
    n = len(records)
    deltas = []
    for _ in range(resamples):
        s = [records[rng.randrange(n)] for _ in range(n)]
        deltas.append(delta(s))
    deltas.sort()

    centred = [d - observed for d in deltas]
    extreme = sum(1 for d in centred if abs(d) >= abs(observed))
    return {
        "delta": round(observed, 4),
        "ci95": (round(deltas[int(0.025 * resamples)], 4),
                 round(deltas[min(resamples - 1, int(0.975 * resamples))], 4)),
        "p_value": round((extreme + 1) / (resamples + 1), 4),
    }


# ── Line-item extraction ───────────────────────────────────────────────────
_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^a-z0-9 ]")


def normalise_name(s: str) -> str:
    s = _PUNCT.sub(" ", str(s).lower())
    return _WS.sub(" ", s).strip()


def name_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, normalise_name(a), normalise_name(b)).ratio()


def match_items(
    gold_items: list[dict],
    pred_items: list[dict],
    name_threshold: float = 0.80,
    amount_tolerance: float = 0.01,
) -> tuple[list[tuple[dict, dict]], list[dict], list[dict]]:
    """
    Greedy one-to-one matching, best pairs first.

    A pair matches when the names are similar enough AND the amounts agree
    within tolerance (relative, floored at 0.01 absolute). Requiring both is
    what keeps this from rewarding a model that hallucinates plausible item
    names with wrong numbers -- the failure mode that actually matters when the
    output is going into a claims pipeline.
    """
    candidates = []
    for gi, g in enumerate(gold_items):
        for pi, p in enumerate(pred_items):
            sim = name_similarity(g.get("item_name", ""), p.get("item_name", ""))
            if sim < name_threshold:
                continue
            ga = float(g.get("item_amount") or 0.0)
            pa = float(p.get("item_amount") or 0.0)
            if abs(ga - pa) > max(amount_tolerance * abs(ga), 0.01):
                continue
            candidates.append((sim, gi, pi))

    candidates.sort(reverse=True)
    used_g, used_p, matched = set(), set(), []
    for _, gi, pi in candidates:
        if gi in used_g or pi in used_p:
            continue
        used_g.add(gi)
        used_p.add(pi)
        matched.append((gold_items[gi], pred_items[pi]))

    missed = [g for i, g in enumerate(gold_items) if i not in used_g]
    spurious = [p for i, p in enumerate(pred_items) if i not in used_p]
    return matched, missed, spurious


def score_extraction(
    gold_pages: list[list[dict]],
    pred_pages: list[list[dict]],
    resamples: int = DEFAULT_RESAMPLES,
) -> dict:
    """Micro-averaged item P/R/F1 plus amount error, with page-level bootstrap."""
    if len(gold_pages) != len(pred_pages):
        raise ValueError("page count mismatch")

    per_page = []
    for g_items, p_items in zip(gold_pages, pred_pages):
        matched, missed, spurious = match_items(g_items, p_items)
        abs_err = [
            abs(float(g.get("item_amount") or 0) - float(p.get("item_amount") or 0))
            for g, p in matched
        ]
        per_page.append(
            {
                "tp": len(matched),
                "fn": len(missed),
                "fp": len(spurious),
                "gold_total": sum(float(i.get("item_amount") or 0) for i in g_items),
                "pred_total": sum(float(i.get("item_amount") or 0) for i in p_items),
                "abs_err_sum": sum(abs_err),
                "abs_err_n": len(abs_err),
            }
        )

    def micro_f1(sample):
        tp = sum(r["tp"] for r in sample)
        fp = sum(r["fp"] for r in sample)
        fn = sum(r["fn"] for r in sample)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        return 2 * prec * rec / (prec + rec) if prec + rec else 0.0

    tp = sum(r["tp"] for r in per_page)
    fp = sum(r["fp"] for r in per_page)
    fn = sum(r["fn"] for r in per_page)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0

    err_n = sum(r["abs_err_n"] for r in per_page)
    mae = sum(r["abs_err_sum"] for r in per_page) / err_n if err_n else 0.0

    gold_tot = sum(r["gold_total"] for r in per_page)
    pred_tot = sum(r["pred_total"] for r in per_page)

    return {
        "n_pages": len(per_page),
        "gold_items": tp + fn,
        "pred_items": tp + fp,
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "micro_f1": round(2 * prec * rec / (prec + rec), 4) if prec + rec else 0.0,
        "micro_f1_ci95": bootstrap_ci(per_page, micro_f1, resamples),
        "amount_mae_on_matched": round(mae, 4),
        "grand_total_gold": round(gold_tot, 2),
        "grand_total_pred": round(pred_tot, 2),
        "grand_total_abs_pct_error": (
            round(abs(pred_tot - gold_tot) / gold_tot * 100, 2) if gold_tot else None
        ),
    }


# ── Latency ────────────────────────────────────────────────────────────────
def percentile(values: list[float], q: float) -> float:
    """Linear-interpolated percentile; q in [0, 100]."""
    if not values:
        return float("nan")
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    pos = (len(s) - 1) * (q / 100.0)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return s[int(pos)]
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def latency_summary(seconds: list[float]) -> dict:
    if not seconds:
        return {}
    return {
        "n": len(seconds),
        "mean_s": round(sum(seconds) / len(seconds), 3),
        "p50_s": round(percentile(seconds, 50), 3),
        "p90_s": round(percentile(seconds, 90), 3),
        "p95_s": round(percentile(seconds, 95), 3),
        "min_s": round(min(seconds), 3),
        "max_s": round(max(seconds), 3),
    }
