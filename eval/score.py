"""
Score extraction output against hand-labelled ground truth.

Why this exists: without it you cannot tell whether a prompt change helped.
Every tweak to SYSTEM_PROMPT is otherwise a guess, and guesses on a 20-item
table are indistinguishable from noise.

Deliberately stdlib-only, so it runs with no dependencies installed.

Workflow:

    # 1. Generate a labelling template from the current extractor output
    python -m eval.score --make-template bill.pdf

    # 2. Open eval/gold/bill.json and CORRECT it by hand. This is the only
    #    step that costs you real time, and it is the step that makes every
    #    number afterwards meaningful.

    # 3. Score current output against it, any time you change the prompt
    python -m eval.score --score

Matching rule: a predicted item matches a gold item when the names are similar
enough AND the amounts agree. Requiring both is deliberate -- a model that
invents plausible item names with wrong numbers is worse than useless in a
claims pipeline, and name-only matching would score it highly.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

GOLD_DIR = Path(__file__).resolve().parent / "gold"
GOLD_DIR.mkdir(parents=True, exist_ok=True)

NAME_THRESHOLD = 0.80
AMOUNT_TOLERANCE_PCT = 0.01

_PUNCT = re.compile(r"[^a-z0-9 ]")
_WS = re.compile(r"\s+")


def normalise(s: str) -> str:
    return _WS.sub(" ", _PUNCT.sub(" ", str(s).lower())).strip()


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, normalise(a), normalise(b)).ratio()


def match_items(gold: list[dict], pred: list[dict]) -> tuple[list, list, list]:
    """
    Greedy one-to-one matching, best pairs first.

    One-to-one matters: bills legitimately repeat identical rows (20 identical
    glucometer charges). Many-to-one matching would let a single predicted row
    satisfy all 20 gold rows and report perfect recall on a broken extraction.
    """
    candidates = []
    for gi, g in enumerate(gold):
        for pi, p in enumerate(pred):
            sim = similarity(g.get("item_name", ""), p.get("item_name", ""))
            if sim < NAME_THRESHOLD:
                continue
            ga = float(g.get("item_amount") or 0.0)
            pa = float(p.get("item_amount") or 0.0)
            if abs(ga - pa) > max(AMOUNT_TOLERANCE_PCT * abs(ga), 0.01):
                continue
            candidates.append((sim, gi, pi))

    candidates.sort(reverse=True)
    used_g, used_p, matched = set(), set(), []
    for _, gi, pi in candidates:
        if gi in used_g or pi in used_p:
            continue
        used_g.add(gi)
        used_p.add(pi)
        matched.append((gold[gi], pred[pi]))

    missed = [g for i, g in enumerate(gold) if i not in used_g]
    spurious = [p for i, p in enumerate(pred) if i not in used_p]
    return matched, missed, spurious


def flatten(result: dict) -> list[dict]:
    """All line items across pages, excluding suppressed Bill Summary pages."""
    items = []
    for page in (result.get("data") or {}).get("pagewise_line_items", []):
        if page.get("page_type") == "Bill Summary":
            continue
        items.extend(page.get("bill_items") or [])
    return items


def score_one(gold_doc: dict, pred_doc: dict) -> dict:
    gold_items = gold_doc.get("bill_items", [])
    pred_items = flatten(pred_doc) if "data" in pred_doc else pred_doc.get("bill_items", [])

    matched, missed, spurious = match_items(gold_items, pred_items)
    tp, fn, fp = len(matched), len(missed), len(spurious)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0

    gold_total = sum(float(i.get("item_amount") or 0) for i in gold_items)
    pred_total = sum(float(i.get("item_amount") or 0) for i in pred_items)

    return {
        "gold_items": len(gold_items),
        "pred_items": len(pred_items),
        "matched": tp,
        "missed": fn,
        "spurious": fp,
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "gold_total": round(gold_total, 2),
        "pred_total": round(pred_total, 2),
        "total_abs_pct_error": (
            round(abs(pred_total - gold_total) / gold_total * 100, 2)
            if gold_total else None
        ),
        "missed_examples": [i.get("item_name") for i in missed[:5]],
        "spurious_examples": [i.get("item_name") for i in spurious[:5]],
    }


def make_template(pdf_path: Path) -> None:
    from app.extractor import BillExtractor

    print(f"extracting {pdf_path.name} ...")
    result = BillExtractor().extract(str(pdf_path)).model_dump()
    items = flatten(result)

    out = GOLD_DIR / f"{pdf_path.stem}.json"
    out.write_text(json.dumps(
        {
            "source": pdf_path.name,
            "_instructions": (
                "CORRECT THIS BY HAND. These are model outputs, not ground "
                "truth. Fix wrong amounts, add missed rows, delete invented "
                "ones. Keep every legitimately repeated row as a separate "
                "entry. Then set verified to true."
            ),
            "verified": False,
            "bill_items": items,
        },
        indent=2, ensure_ascii=False,
    ))
    print(f"template -> {out}  ({len(items)} items to review)")


def run_scoring(pdf_dir: Path) -> int:
    from app.extractor import BillExtractor

    golds = sorted(GOLD_DIR.glob("*.json"))
    if not golds:
        raise SystemExit(
            f"No gold files in {GOLD_DIR}. Create one with --make-template first."
        )

    extractor = BillExtractor()
    rows, unverified = [], []

    for gold_path in golds:
        gold = json.loads(gold_path.read_text())
        if not gold.get("verified"):
            unverified.append(gold_path.name)
            continue

        pdf = pdf_dir / gold["source"]
        if not pdf.exists():
            print(f"  skip {gold['source']}: not found in {pdf_dir}")
            continue

        pred = extractor.extract(str(pdf)).model_dump()
        r = score_one(gold, pred)
        r["source"] = gold["source"]
        recon = (pred.get("data") or {}).get("reconciliation") or {}
        r["reconciles"] = recon.get("matches")
        rows.append(r)
        print(f"  {gold['source']:<28} F1 {r['f1']:.3f}  "
              f"(+{r['matched']} -{r['missed']} ~{r['spurious']})  "
              f"total err {r['total_abs_pct_error']}%")

    if unverified:
        print(f"\nSkipped {len(unverified)} unverified template(s): "
              f"{', '.join(unverified)}")
        print("Set verified=true once you have corrected them by hand.")

    if not rows:
        print("\nNothing scored.")
        return 1

    tp = sum(r["matched"] for r in rows)
    fn = sum(r["missed"] for r in rows)
    fp = sum(r["spurious"] for r in rows)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0

    print(f"\n{'='*58}\nMICRO-AVERAGED over {len(rows)} document(s)")
    print(f"  precision {prec:.3f}   recall {rec:.3f}   F1 {f1:.3f}")
    print(f"  matched {tp}, missed {fn}, spurious {fp}")

    # Does the label-free reconciliation signal agree with real accuracy? If it
    # does, you can trust it on unlabelled documents in production.
    agree = [r for r in rows if r["reconciles"] is not None]
    if agree:
        good = [r for r in agree if r["reconciles"]]
        bad = [r for r in agree if not r["reconciles"]]
        print(f"\n  reconciliation passed on {len(good)}, failed on {len(bad)}")
        if good and bad:
            gf = sum(r["f1"] for r in good) / len(good)
            bf = sum(r["f1"] for r in bad) / len(bad)
            print(f"    mean F1 when it passes: {gf:.3f}")
            print(f"    mean F1 when it fails:  {bf:.3f}")
            print("    (a clear gap means reconciliation is a usable proxy for "
                  "accuracy on unlabelled bills)")

    out = Path(__file__).resolve().parent / "results.json"
    out.write_text(json.dumps(
        {"micro": {"precision": round(prec, 4), "recall": round(rec, 4),
                   "f1": round(f1, 4), "documents": len(rows)},
         "per_document": rows}, indent=2))
    print(f"\n-> {out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--make-template", type=Path, help="PDF to bootstrap a gold file from")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--pdf-dir", type=Path, default=REPO_ROOT / "samples")
    args = ap.parse_args()

    if args.make_template:
        make_template(args.make_template)
        return 0
    if args.score:
        return run_scoring(args.pdf_dir)
    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
