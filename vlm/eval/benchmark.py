"""
Head-to-head benchmark: LoRA-adapted Qwen2-VL-2B vs the Gemini baseline.

Produces artifacts/results/benchmark_<timestamp>.json and a markdown summary.
Every accuracy figure carries a bootstrap CI and every cost figure carries its
volume assumption. Nothing is reported that was not measured in this run.

Guard rails, on purpose:
  * refuses to run until the held-out labels are human-verified
  * refuses to report a delta as significant when the paired CI straddles zero
  * times only the inference call, after warmup

Usage:
    python -m vlm.eval.benchmark --backends local gemini --adapter stage_c
    python -m vlm.eval.benchmark --backends local --adapter stage_c --task classify
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from vlm.config import LABELS_DIR, MODEL, PAGE_TYPES, RESULTS_DIR  # noqa: E402
from vlm.eval import cost_model  # noqa: E402
from vlm.eval.metrics import (  # noqa: E402
    latency_summary,
    paired_bootstrap_delta,
    score_classification,
    score_extraction,
)


def load_verified_testset() -> list[dict]:
    flag = LABELS_DIR / "VERIFIED"
    review = LABELS_DIR / "test_review.csv"
    raw = LABELS_DIR / "test_raw.jsonl"

    for p in (flag, review, raw):
        if not p.exists():
            raise SystemExit(f"{p} missing. Run `python -m vlm.data.distill` first.")

    if not json.loads(flag.read_text()).get("verified"):
        raise SystemExit(
            "Held-out labels are not human-verified.\n\n"
            "  Benchmarking against unreviewed teacher output measures agreement\n"
            "  with Gemini, not accuracy, and caps the student at 100% by\n"
            "  construction. Correct `corrected_page_type` in\n"
            f"  {review}\n"
            f"  then set verified=true in {flag}."
        )

    corrections = {}
    with review.open() as f:
        for row in csv.DictReader(f):
            corrections[row["page_id"]] = row["corrected_page_type"].strip()

    pages = []
    for line in raw.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        gold_type = corrections.get(r["page_id"], r["page_type"])
        if gold_type not in PAGE_TYPES:
            raise SystemExit(
                f"page {r['page_id']}: corrected_page_type {gold_type!r} is not one of {PAGE_TYPES}"
            )
        r["gold_page_type"] = gold_type
        pages.append(r)
    return pages


def run_local(pages: list[dict], adapter: str, task: str) -> dict:
    from vlm.serve.local_vlm import LocalVLM

    vlm = LocalVLM(adapter=adapter)
    vlm.warmup(pages[0]["abs_path"])

    cls_pred, ext_pred, cls_lat, ext_lat = [], [], [], []
    for i, p in enumerate(pages, 1):
        if task in ("classify", "both"):
            label, el = vlm.classify(p["abs_path"])
            cls_pred.append(label)
            cls_lat.append(el)
        if task in ("extract", "both"):
            data, el = vlm.extract(p["abs_path"])
            ext_pred.append(data["bill_items"])
            ext_lat.append(el)
        print(f"  [{i}/{len(pages)}] {p['page_id']}")

    return {
        "backend": "local",
        "adapter": vlm.adapter_path,
        "base_model": MODEL.model_id,
        "load_seconds": vlm.load_seconds,
        "cls_pred": cls_pred,
        "ext_pred": ext_pred,
        "cls_latency": cls_lat,
        "ext_latency": ext_lat,
        "input_tokens": 0,
        "output_tokens": 0,
    }


def run_gemini(pages: list[dict], task: str) -> dict:
    from app.extractor import GeminiCaller, _encode_image, _ocr_page, _parse_page_result
    from PIL import Image

    cls_pred, ext_pred, lat = [], [], []
    tot_in = tot_out = 0

    for i, p in enumerate(pages, 1):
        caller = GeminiCaller()
        try:
            hint = _ocr_page(Image.open(p["abs_path"]))
        except Exception:
            hint = ""
        b64 = _encode_image(p["abs_path"])

        t0 = time.perf_counter()
        raw = caller.call([b64], [hint], [1])
        lat.append(time.perf_counter() - t0)

        page, _ = _parse_page_result(raw[0], 1)
        cls_pred.append(page.page_type if page.page_type in PAGE_TYPES else "Other")
        ext_pred.append([it.model_dump() for it in page.bill_items])
        tot_in += caller.token_usage.input_tokens
        tot_out += caller.token_usage.output_tokens
        print(f"  [{i}/{len(pages)}] {p['page_id']}")

    return {
        "backend": "gemini",
        "model": MODEL.model_id,
        "cls_pred": cls_pred,
        "ext_pred": ext_pred,
        "cls_latency": lat,
        "ext_latency": lat,  # one call yields both; latency is not separable
        "input_tokens": tot_in,
        "output_tokens": tot_out,
    }


def markdown_report(report: dict) -> str:
    L = []
    A = L.append
    A("# Benchmark: LoRA Qwen2-VL-2B vs Gemini baseline\n")
    A(f"Run: `{report['run_id']}`  |  pages: {report['n_pages']}  "
      f"|  documents: {report['n_documents']}\n")

    A("\n## Accuracy\n")
    A("| backend | task | metric | value | 95% CI |")
    A("|---|---|---|---|---|")
    for b, res in report["results"].items():
        if "classification" in res:
            c = res["classification"]
            A(f"| {b} | classify | accuracy | {c['accuracy']:.3f} | "
              f"[{c['accuracy_ci95'][0]:.3f}, {c['accuracy_ci95'][1]:.3f}] |")
            A(f"| {b} | classify | macro-F1 | {c['macro_f1']:.3f} | "
              f"[{c['macro_f1_ci95'][0]:.3f}, {c['macro_f1_ci95'][1]:.3f}] |")
        if "extraction" in res:
            e = res["extraction"]
            A(f"| {b} | extract | item micro-F1 | {e['micro_f1']:.3f} | "
              f"[{e['micro_f1_ci95'][0]:.3f}, {e['micro_f1_ci95'][1]:.3f}] |")
            A(f"| {b} | extract | amount MAE (matched) | {e['amount_mae_on_matched']:.2f} | — |")

    mb = report["results"].get("majority_baseline", {}).get("classification")
    if mb:
        A(f"\n**Majority-class baseline** (always predict "
          f"`{mb['predicted_class']}`): accuracy **{mb['accuracy']:.3f}**. "
          f"Any backend at or below this has learned nothing useful.\n")

    if report.get("paired_delta"):
        d = report["paired_delta"]
        A("\n## Paired comparison (local − gemini, classification accuracy)\n")
        A(f"- delta: **{d['delta']:+.3f}**  (95% CI [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}], p={d['p_value']:.3f})")
        straddles = d["ci95"][0] <= 0 <= d["ci95"][1]
        A(f"- **{'NOT statistically distinguishable' if straddles else 'Statistically distinguishable'}** "
          f"at this sample size.")
        if straddles:
            A("- Do not claim a specific accuracy gap from this run. The honest "
              "statement is that the difference is within noise for n="
              f"{report['n_pages']}.")

    A("\n## Latency\n")
    A("| backend | task | p50 | p90 | p95 | mean |")
    A("|---|---|---|---|---|---|")
    for b, res in report["results"].items():
        for t in ("classify", "extract"):
            lt = res.get("latency", {}).get(t)
            if lt:
                A(f"| {b} | {t} | {lt['p50_s']:.2f}s | {lt['p90_s']:.2f}s | "
                  f"{lt['p95_s']:.2f}s | {lt['mean_s']:.2f}s |")

    if report.get("cost"):
        c = report["cost"]
        A("\n## Cost\n")
        A(f"- Gemini measured: {c['api']['input_tokens']:,} in / "
          f"{c['api']['output_tokens']:,} out tokens over {c['api']['pages']} pages "
          f"= **${c['api']['usd_per_1000_pages']:.3f} / 1000 pages** "
          f"(list price as of {c['api']['price_as_of']})")
        A(f"- Local on {c['local']['deployment']}: "
          f"**${c['local']['marginal_usd_per_1000_pages']:.3f} / 1000 pages** marginal, "
          f"plus ${c['local']['training_usd']:.2f} one-off training")
        be = c["breakeven_pages"]
        A(f"- Breakeven: **{f'{be:,} pages' if be is not None else 'never on this target'}** "
          f"— {c['breakeven_note']}")
        A("\n| volume | API | local | cheaper |")
        A("|---|---|---|---|")
        for vol, p in c["projections"].items():
            A(f"| {vol.replace('_', ' ')} | ${p['api_usd']:,.2f} | ${p['local_usd']:,.2f} | {p['cheaper']} |")
        A("\nCaveats:")
        for cv in c["caveats"]:
            A(f"- {cv}")

    A("\n## Environment\n")
    for k, v in report["environment"].items():
        A(f"- {k}: `{v}`")
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backends", nargs="+", default=["local", "gemini"],
                    choices=["local", "gemini"])
    ap.add_argument("--adapter", default="stage_c")
    ap.add_argument("--task", default="both", choices=["classify", "extract", "both"])
    ap.add_argument("--deployment", default="T4", help="cost target: T4|A10G|A100-40GB|on-device")
    ap.add_argument("--training-gpu-hours", type=float, default=0.0)
    args = ap.parse_args()

    pages = load_verified_testset()
    gold_cls = [p["gold_page_type"] for p in pages]
    gold_ext = [p["bill_items"] for p in pages]

    print(f"{len(pages)} held-out pages, {len({p['source_pdf'] for p in pages})} documents")
    if len(pages) < 30:
        print(f"  NOTE: n={len(pages)}. Confidence intervals will be wide; that is "
              f"the finding, not a defect.")

    raws, results = {}, {}
    for b in args.backends:
        print(f"\n-- {b} --")
        raw = run_local(pages, args.adapter, args.task) if b == "local" else run_gemini(pages, args.task)
        raws[b] = raw

        res: dict = {"latency": {}}
        if args.task in ("classify", "both"):
            res["classification"] = score_classification(
                gold_cls, raw["cls_pred"], PAGE_TYPES
            ).to_dict()
            res["latency"]["classify"] = latency_summary(raw["cls_latency"])
        if args.task in ("extract", "both"):
            # Gemini's own output is the extraction reference, so scoring Gemini
            # against it is vacuous. We keep the row only for the local model and
            # label the reference honestly in the report.
            if b == "local":
                res["extraction"] = score_extraction(gold_ext, raw["ext_pred"])
            res["latency"]["extract"] = latency_summary(raw["ext_latency"])
        results[b] = res

    # Majority-class baseline. The corpus is dominated by Bill Detail pages, so
    # a model that always guesses the majority class already scores well. Any
    # accuracy claim that isn't clearly above this line is meaningless, and
    # reporting it is the difference between an evaluation and a demo.
    majority = None
    if args.task in ("classify", "both") and gold_cls:
        top = max(set(gold_cls), key=gold_cls.count)
        majority = score_classification(gold_cls, [top] * len(gold_cls), PAGE_TYPES).to_dict()
        majority["predicted_class"] = top
        results["majority_baseline"] = {"classification": majority, "latency": {}}

    paired = None
    if {"local", "gemini"} <= set(args.backends) and args.task in ("classify", "both"):
        paired = paired_bootstrap_delta(gold_cls, raws["local"]["cls_pred"], raws["gemini"]["cls_pred"])

    cost = None
    if "gemini" in raws and "local" in raws:
        lat = raws["local"]["ext_latency"] or raws["local"]["cls_latency"]
        cost = cost_model.compare(
            pages=len(pages),
            api_input_tokens=raws["gemini"]["input_tokens"],
            api_output_tokens=raws["gemini"]["output_tokens"],
            local_seconds_per_page=sum(lat) / len(lat) if lat else 0.0,
            deployment=args.deployment,
            training_gpu_hours=args.training_gpu_hours,
        )

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report = {
        "run_id": run_id,
        "n_pages": len(pages),
        "n_documents": len({p["source_pdf"] for p in pages}),
        "task": args.task,
        "results": results,
        "paired_delta": paired,
        "cost": cost,
        "extraction_reference": (
            "Gemini teacher output (page_type hand-corrected). Extraction "
            "line-items are NOT hand-verified; item F1 measures agreement with "
            "the teacher, not ground truth."
        ),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "base_model": MODEL.model_id,
            "max_pixels": MODEL.max_pixels,
            "load_in_4bit": MODEL.load_in_4bit,
        },
    }

    json_path = RESULTS_DIR / f"benchmark_{run_id}.json"
    md_path = RESULTS_DIR / f"benchmark_{run_id}.md"
    json_path.write_text(json.dumps(report, indent=2))
    md_path.write_text(markdown_report(report))

    print(f"\n{json_path}\n{md_path}")
    print("\n" + markdown_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
