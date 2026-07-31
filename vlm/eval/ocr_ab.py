"""
Measure the OCR stage: does the hint earn its latency, and did the fixes work?

Two modes:

  --mode ocr   Offline. No API key, no cost. Times the OCR stage, reports how
               much text the old 1500-char cap discarded, shows which pages get
               their hint dropped as unreliable, and scores accuracy against
               pdftotext ground truth on the files that have a text layer.

  --mode ab    Runs the full Gemini extraction twice per page, with
               USE_OCR_HINT=true and false, and compares the extracted line
               items. This costs API calls. It answers the actual question:
               with the page image already in the prompt, does the OCR hint
               change the output enough to justify 1-2s/page?

Usage:
    python -m vlm.eval.ocr_ab --mode ocr
    python -m vlm.eval.ocr_ab --mode ab --limit 10
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from vlm.config import PAGES_DIR, RESULTS_DIR  # noqa: E402

WORD = re.compile(r"[a-z0-9][a-z0-9.,/-]*")
NUM = re.compile(r"\d[\d,]*\.?\d*")
OLD_CAP = 1500  # the value this pipeline used before the fix


def _f1(pred: list[str], gold: list[str]) -> float:
    p, g = Counter(pred), Counter(gold)
    overlap = sum((p & g).values())
    if not overlap:
        return 0.0
    prec = overlap / max(sum(p.values()), 1)
    rec = overlap / max(sum(g.values()), 1)
    return 2 * prec * rec / (prec + rec)


def _truth_for(page: dict, pdf_dir: Path) -> str | None:
    """pdftotext output for a page, when the source PDF has a real text layer."""
    pdf = pdf_dir / page["source_pdf"]
    if not pdf.exists():
        return None
    try:
        out = subprocess.run(
            ["pdftotext", "-layout", "-f", str(page["page_no"]),
             "-l", str(page["page_no"]), str(pdf), "-"],
            capture_output=True, text=True, timeout=60,
        ).stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    # A junk OCR text layer looks like text but scores near zero; the caller
    # still reports it, flagged, rather than silently trusting it.
    return out if len(out.strip()) > 200 else None


def mode_ocr(pages: list[dict], pdf_dir: Path) -> dict:
    from PIL import Image

    from app.extractor import OCR_MIN_CONFIDENCE, _autorotate, _ocr_page

    rows, lat = [], []
    print(f"{'page':<26} {'sec':>5} {'rot':>4} {'chars':>6} {'lost@1500':>10} {'tokF1':>7} {'numF1':>7}")
    print("-" * 76)

    for p in pages:
        img = Image.open(p["abs_path"])
        t0 = time.perf_counter()
        img, deg = _autorotate(img)
        text = _ocr_page(img)
        el = time.perf_counter() - t0
        lat.append(el)

        n = len(text.strip())
        lost = round(100 * max(0, n - OLD_CAP) / n, 1) if n else 0.0

        tok_f1 = num_f1 = None
        truth = _truth_for(p, pdf_dir)
        if truth:
            tl, tn = WORD.findall(truth.lower()), NUM.findall(truth)
            tok_f1 = round(_f1(WORD.findall(text.lower()), tl) * 100, 1)
            num_f1 = round(_f1(NUM.findall(text), tn) * 100, 1)

        rows.append({
            "page_id": p["page_id"], "seconds": round(el, 3), "rotation": deg,
            "chars": n, "pct_lost_at_old_1500_cap": lost,
            "token_f1": tok_f1, "number_f1": num_f1,
            "hint_dropped": n == 0,
        })
        print(f"{p['page_id']:<26} {el:>5.2f} {deg:>4} {n:>6} "
              f"{lost:>9.1f}% {('-' if tok_f1 is None else f'{tok_f1:.1f}%'):>7} "
              f"{('-' if num_f1 is None else f'{num_f1:.1f}%'):>7}")

    scored = [r for r in rows if r["number_f1"] is not None]
    dropped = [r for r in rows if r["hint_dropped"]]
    rotated = [r for r in rows if r["rotation"]]
    truncated = [r for r in rows if r["pct_lost_at_old_1500_cap"] > 0]

    summary = {
        "pages": len(rows),
        "total_ocr_seconds": round(sum(lat), 1),
        "mean_seconds_per_page": round(sum(lat) / len(lat), 3) if lat else 0,
        "pages_rotated": len(rotated),
        "pages_hint_dropped_low_confidence": len(dropped),
        "min_confidence_threshold": OCR_MIN_CONFIDENCE,
        "pages_that_would_have_been_truncated": len(truncated),
        "mean_pct_lost_at_old_cap": (
            round(sum(r["pct_lost_at_old_1500_cap"] for r in truncated) / len(truncated), 1)
            if truncated else 0.0
        ),
        "pages_with_ground_truth": len(scored),
        "mean_token_f1": (
            round(sum(r["token_f1"] for r in scored) / len(scored), 1) if scored else None
        ),
        "mean_number_f1": (
            round(sum(r["number_f1"] for r in scored) / len(scored), 1) if scored else None
        ),
        "per_page": rows,
    }

    print(f"\n{summary['pages']} pages in {summary['total_ocr_seconds']}s "
          f"({summary['mean_seconds_per_page']}s/page)")
    print(f"  rotation corrected on {len(rotated)} page(s)")
    print(f"  hint dropped as unreliable on {len(dropped)} page(s)")
    print(f"  old 1500-char cap would have truncated {len(truncated)} page(s), "
          f"mean {summary['mean_pct_lost_at_old_cap']}% of the hint discarded")
    if scored:
        print(f"  vs pdftotext ground truth on {len(scored)} page(s): "
              f"token F1 {summary['mean_token_f1']}%, number F1 {summary['mean_number_f1']}%")
    else:
        print("  no pages with a usable text layer -- accuracy not measurable here")
    return summary


def mode_ab(pages: list[dict], limit: int) -> dict:
    """Same pages, hint on vs off. Requires GOOGLE_API_KEY and costs money."""
    if not os.getenv("GOOGLE_API_KEY"):
        raise SystemExit("GOOGLE_API_KEY not set")

    import importlib

    from PIL import Image

    rows = []
    for p in pages[:limit] if limit else pages:
        result = {}
        for use_hint in (True, False):
            os.environ["USE_OCR_HINT"] = "true" if use_hint else "false"
            # Config is read at import time, so the module must be reloaded for
            # the flag to take effect.
            import app.extractor as ex
            importlib.reload(ex)

            img = Image.open(p["abs_path"])
            img, _ = ex._autorotate(img)
            hint = ex._ocr_page(img) if use_hint else ""
            b64 = ex._encode_image(p["abs_path"])

            caller = ex.GeminiCaller()
            t0 = time.perf_counter()
            raw = caller.call([b64], [hint], [1])
            elapsed = time.perf_counter() - t0
            page, _ = ex._parse_page_result(raw[0], 1)

            result["on" if use_hint else "off"] = {
                "page_type": page.page_type,
                "n_items": len(page.bill_items),
                "total": round(sum(i.item_amount for i in page.bill_items), 2),
                "seconds": round(elapsed, 2),
                "input_tokens": caller.token_usage.input_tokens,
            }

        on, off = result["on"], result["off"]
        row = {
            "page_id": p["page_id"], **{f"hint_{k}": v for k, v in result.items()},
            "item_delta": on["n_items"] - off["n_items"],
            "total_delta": round(on["total"] - off["total"], 2),
            "type_agrees": on["page_type"] == off["page_type"],
        }
        rows.append(row)
        print(f"{p['page_id']:<26} on={on['n_items']:>3} items/{on['total']:>10.2f}  "
              f"off={off['n_items']:>3} items/{off['total']:>10.2f}  "
              f"{'type OK' if row['type_agrees'] else 'TYPE DIFFERS'}")

    same_items = sum(1 for r in rows if r["item_delta"] == 0)
    same_total = sum(1 for r in rows if abs(r["total_delta"]) < 0.01)
    print(f"\n{len(rows)} pages: identical item count on {same_items}, "
          f"identical total on {same_total}")
    print("If those are near-identical, the OCR hint is not earning its "
          "latency and you can drop it.")
    return {"pages": len(rows), "identical_item_count": same_items,
            "identical_total": same_total, "per_page": rows}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["ocr", "ab"], default="ocr")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--pdf-dir", type=Path, default=Path("/tmp/pdfs"))
    args = ap.parse_args()

    manifest_path = PAGES_DIR / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"{manifest_path} missing. Run `python -m vlm.data.render` first.")
    pages = json.loads(manifest_path.read_text())
    if args.limit:
        pages = pages[: args.limit]

    out = mode_ocr(pages, args.pdf_dir) if args.mode == "ocr" else mode_ab(pages, args.limit)

    dest = RESULTS_DIR / f"ocr_{args.mode}.json"
    dest.write_text(json.dumps(out, indent=2))
    print(f"\n-> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
