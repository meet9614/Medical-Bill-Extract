"""
Distil the Gemini teacher into labels for the medical pages.

Methodology note -- read this before trusting any number downstream:

    Labels produced here are *teacher outputs*, not ground truth. If we both
    train on them and evaluate against them, the benchmark measures agreement
    with Gemini and structurally caps the student at 100% "accuracy" while
    inheriting every teacher error invisibly.

    So this script splits the pages first, then emits two files:

      train_distilled.jsonl  -- teacher labels, used for training. Noise here is
                                acceptable; it is what distillation is.
      test_review.csv        -- the held-out split, pre-filled with the teacher's
                                guess, for YOU to correct by hand.

    The test split must be human-verified before benchmark.py will consume it.
    The script writes a `verified: false` flag and the evaluator refuses to run
    until you flip it. This is the difference between "my model agrees with
    Gemini 94% of the time" and "my model is 94% accurate" -- an interviewer
    will ask, and only one of those is a real claim.

Usage:
    python -m vlm.data.distill --limit 0        # all pages
    python -m vlm.data.distill --dry-run        # cost estimate, no API calls
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from vlm.config import (  # noqa: E402
    GEMINI_INPUT_USD_PER_MTOK,
    GEMINI_OUTPUT_USD_PER_MTOK,
    LABELS_DIR,
    PAGE_TYPES,
    PAGES_DIR,
    SEED,
    TEST_FRACTION,
    group_key,
)


def load_manifest() -> list[dict]:
    mpath = PAGES_DIR / "manifest.json"
    if not mpath.exists():
        raise SystemExit(
            f"{mpath} missing. Run `python -m vlm.data.render` first."
        )
    return json.loads(mpath.read_text())


def split_pages(manifest: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    Split by DOCUMENT GROUP, not by page and not by file.

    Pages from one bill share a template, a hospital, and often a scanner. A
    page-level split leaks all of that across the boundary and inflates test
    accuracy substantially.

    Grouping by *file* is also not enough here: two of the supplied samples are
    contiguous slices of the same 90-page bill. config.group_key() collapses
    known-related files into one splittable unit.
    """
    groups = sorted({group_key(m["source_pdf"]) for m in manifest})
    rng = random.Random(SEED)
    rng.shuffle(groups)
    n_test = max(1, round(len(groups) * TEST_FRACTION))
    test_groups = set(groups[:n_test])

    train = [m for m in manifest if group_key(m["source_pdf"]) not in test_groups]
    test = [m for m in manifest if group_key(m["source_pdf"]) in test_groups]
    return train, test


def call_teacher(image_path: str) -> tuple[dict, int, int]:
    """Run one page through the existing Gemini pipeline. Returns (page, in_tok, out_tok)."""
    from app.extractor import GeminiCaller, _encode_image, _ocr_page, _parse_page_result

    try:
        from PIL import Image

        ocr_hint = _ocr_page(Image.open(image_path))
    except Exception:
        ocr_hint = ""

    caller = GeminiCaller()
    b64 = _encode_image(image_path)
    raw_list = caller.call([b64], [ocr_hint], [1])
    page, fraud = _parse_page_result(raw_list[0], 1)
    usage = caller.token_usage

    return (
        {
            "page_type": page.page_type,
            "bill_items": [i.model_dump() for i in page.bill_items],
            "fraud_flags": fraud,
        },
        usage.input_tokens,
        usage.output_tokens,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="0 = all pages")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--sleep", type=float, default=1.0, help="seconds between calls")
    args = ap.parse_args()

    manifest = load_manifest()
    train, test = split_pages(manifest)

    n_groups = len({group_key(m["source_pdf"]) for m in manifest})
    print(f"{len(manifest)} pages / {len({m['source_pdf'] for m in manifest})} files "
          f"/ {n_groups} independent documents")
    print(f"  train: {len(train)} pages / {len({group_key(m['source_pdf']) for m in train})} docs")
    print(f"  test:  {len(test)} pages / {len({group_key(m['source_pdf']) for m in test})} docs")

    if len(test) < 30:
        print(
            f"\n  WARNING: test split is {len(test)} pages. A classification\n"
            f"  accuracy measured on this has a 95% CI of roughly +/-15-25 points.\n"
            f"  Report it as a pilot with the interval attached, never as a bare\n"
            f"  point estimate. The public-data stages exist to give you a second,\n"
            f"  larger held-out set to quote alongside this one.\n"
        )

    if args.dry_run:
        # ~1.3k tokens/page image at our pixel budget, plus OCR hint and prompt.
        est_in = len(manifest) * 2_000
        est_out = len(manifest) * 700
        cost = (
            est_in / 1e6 * GEMINI_INPUT_USD_PER_MTOK
            + est_out / 1e6 * GEMINI_OUTPUT_USD_PER_MTOK
        )
        print(f"\ndry run: ~{est_in:,} in / ~{est_out:,} out tokens, ~${cost:.4f}")
        return 0

    if not os.getenv("GOOGLE_API_KEY"):
        print("error: GOOGLE_API_KEY not set", file=sys.stderr)
        return 1

    pages = manifest if args.limit == 0 else manifest[: args.limit]
    test_ids = {m["page_id"] for m in test}

    train_rows, test_rows = [], []
    tot_in = tot_out = 0

    for i, m in enumerate(pages, 1):
        try:
            label, n_in, n_out = call_teacher(m["abs_path"])
        except Exception as e:  # noqa: BLE001
            print(f"  [{i}/{len(pages)}] {m['page_id']}: FAILED {e}", file=sys.stderr)
            continue
        tot_in += n_in
        tot_out += n_out

        row = {**m, **label}
        (test_rows if m["page_id"] in test_ids else train_rows).append(row)
        print(
            f"  [{i}/{len(pages)}] {m['page_id']}: "
            f"{label['page_type']}, {len(label['bill_items'])} items"
        )
        time.sleep(args.sleep)

    LABELS_DIR.mkdir(parents=True, exist_ok=True)

    train_path = LABELS_DIR / "train_distilled.jsonl"
    with train_path.open("w") as f:
        for r in train_rows:
            f.write(json.dumps(r) + "\n")

    # Held-out split goes out as a review CSV, not as labels.
    review_path = LABELS_DIR / "test_review.csv"
    with review_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["page_id", "source_pdf", "page_no", "teacher_page_type",
             "corrected_page_type", "teacher_item_count", "corrected_item_count",
             "notes"]
        )
        for r in test_rows:
            w.writerow(
                [r["page_id"], r["source_pdf"], r["page_no"], r["page_type"],
                 r["page_type"], len(r["bill_items"]), len(r["bill_items"]), ""]
            )

    (LABELS_DIR / "test_raw.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in test_rows)
    )
    (LABELS_DIR / "VERIFIED").write_text(
        json.dumps({"verified": False, "note": "flip to true after reviewing test_review.csv"})
    )

    cost = tot_in / 1e6 * GEMINI_INPUT_USD_PER_MTOK + tot_out / 1e6 * GEMINI_OUTPUT_USD_PER_MTOK
    print(f"\ntrain labels -> {train_path} ({len(train_rows)} pages)")
    print(f"REVIEW ME    -> {review_path} ({len(test_rows)} pages)")
    print(f"teacher cost: {tot_in:,} in / {tot_out:,} out tokens = ${cost:.4f}")
    print(f"valid page_type values: {PAGE_TYPES}")
    print(
        "\nNext: open test_review.csv, correct `corrected_page_type` by eye,\n"
        "then set verified=true in artifacts/labels/VERIFIED."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
