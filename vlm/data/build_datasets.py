"""
Build unified chat-format datasets for Qwen2-VL fine-tuning.

Three stages, each emitting the same JSONL schema so one training script and one
collator handle all of them:

    {"image": "<abs path>", "task": "classify"|"extract",
     "prompt": "<user turn>", "target": "<assistant turn>"}

  Stage A  rvl_cdip   -- 16-class document classification, thousands of examples.
                        Public, standard, and large enough to actually move the
                        adapter. Teaches the *task shape*: page image in, one
                        class label out.
  Stage B  cord       -- CORD-v2 receipts with real line-item annotations
                        (name / unit price / quantity / total). The closest
                        public analogue to a medical bill's item table, and the
                        only stage where extraction gets a defensible n.
  Stage C  medical    -- your 50 distilled pages. Small. This is domain
                        adaptation on top of A/B, not training from scratch.

Why not train Stage C alone: 35 training pages cannot teach a 2B model a new
output format AND a new domain. A and B buy the format; C buys the domain.

Usage:
    python -m vlm.data.build_datasets --stage rvl_cdip --per-class 400
    python -m vlm.data.build_datasets --stage cord
    python -m vlm.data.build_datasets --stage medical
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from vlm.config import (  # noqa: E402
    DATASET_DIR,
    LABELS_DIR,
    PAGE_TYPES,
    RVL_CDIP_CLASSES,
    SEED,
)

CLASSIFY_PROMPT = (
    "Classify this document page into exactly one of the following types:\n"
    "{classes}\n\n"
    "Respond with the type name only, no other text."
)

EXTRACT_PROMPT = (
    "Extract every line item from this bill page as JSON.\n"
    'Schema: {{"page_type": "<type>", "bill_items": '
    '[{{"item_name": str, "item_amount": float, "item_rate": float|null, '
    '"item_quantity": float|null}}]}}\n'
    "Rules: include every row of every table; exclude subtotals and grand "
    "totals; strip currency symbols; amounts are numbers not strings.\n"
    "Respond with JSON only."
)


def _write(rows: list[dict], name: str) -> Path:
    out = DATASET_DIR / f"{name}.jsonl"
    with out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"{name}: {len(rows)} examples -> {out}")
    return out


# ── Stage A ────────────────────────────────────────────────────────────────
def _open_rvl_cdip():
    """
    Open RVL-CDIP without touching a dataset loading script.

    `aharley/rvl_cdip` ships a Python loading script. That path is a dead end:

      * datasets >= 3.0 removed script-based loading entirely, and
      * pinning back to datasets 2.x breaks against modern huggingface_hub,
        whose URI parser rejects the bare repo id the script uses internally
        ("Repository id must be 'namespace/name', got 'rvl_cdip'").

    So there is no version pair where the script works. Patching the cached
    script is treating the symptom. Instead we load Parquet directly -- HF
    auto-converts every dataset to Parquet on the `refs/convert/parquet`
    branch, which needs no script and no trust_remote_code.

    Override with RVL_CDIP_REPO if you mirror it somewhere else.
    """
    from datasets import load_dataset

    repo = os.getenv("RVL_CDIP_REPO", "aharley/rvl_cdip")
    attempts = [
        ({"path": repo, "split": "train", "streaming": True}, "native parquet"),
        ({"path": repo, "split": "train", "streaming": True,
          "revision": "refs/convert/parquet"}, "auto-converted parquet branch"),
    ]

    errors = []
    for kwargs, label in attempts:
        try:
            ds = load_dataset(**kwargs)
            print(f"  loaded {repo} via {label}")
            return ds
        except Exception as e:  # noqa: BLE001
            errors.append(f"    {label}: {type(e).__name__}: {str(e)[:160]}")

    raise SystemExit(
        f"Could not open {repo} without a loading script.\n"
        + "\n".join(errors)
        + "\n\n"
        "  Stage A is OPTIONAL. It exists to teach the adapter the task shape\n"
        "  (page image in, class label out) on data large enough to move\n"
        "  weights. Stage B (CORD) already teaches the JSON extraction format.\n\n"
        "  To proceed without it, skip straight to stage B and train stage_b\n"
        "  with no --init, then stage C on top:\n\n"
        "    python -m vlm.data.build_datasets --stage cord\n"
        "    python -m vlm.train.train_lora --data stage_b_cord_train --out stage_b --epochs 2\n"
        "    python -m vlm.train.train_lora --data stage_c_medical_train --out stage_c "
        "--init stage_b --epochs 6 --lr 5e-5\n\n"
        "  Then say so in the report: classification had no large-scale\n"
        "  pretraining stage, only 32 medical pages. That is a real limitation\n"
        "  and reviewers respect it stated plainly.\n\n"
        "  Or point RVL_CDIP_REPO at a parquet mirror you trust."
    )


def build_rvl_cdip(per_class: int, image_dir: Path) -> None:
    image_dir.mkdir(parents=True, exist_ok=True)
    prompt = CLASSIFY_PROMPT.format(classes="\n".join(f"- {c}" for c in RVL_CDIP_CLASSES))

    # Streamed: the full corpus is ~400k images / tens of GB and Colab's disk
    # will not hold it. We take a balanced slice instead.
    ds = _open_rvl_cdip()
    counts = {i: 0 for i in range(len(RVL_CDIP_CLASSES))}
    rows, seen = [], 0

    for ex in ds:
        label = int(ex["label"])
        if counts[label] >= per_class:
            if all(c >= per_class for c in counts.values()):
                break
            continue
        seen += 1
        img_path = image_dir / f"rvl_{label:02d}_{counts[label]:05d}.png"
        ex["image"].convert("RGB").save(img_path)
        rows.append(
            {
                "image": str(img_path),
                "task": "classify",
                "prompt": prompt,
                "target": RVL_CDIP_CLASSES[label],
                "source": "rvl_cdip",
            }
        )
        counts[label] += 1
        if seen % 200 == 0:
            print(f"  ...{seen} kept")

    random.Random(SEED).shuffle(rows)
    split = int(len(rows) * 0.9)
    _write(rows[:split], "stage_a_rvlcdip_train")
    _write(rows[split:], "stage_a_rvlcdip_eval")


# ── Stage B ────────────────────────────────────────────────────────────────
def _cord_items(gt: dict) -> list[dict]:
    """CORD-v2 packs its annotation into a JSON string under `gt_parse`."""
    items = []
    menu = gt.get("gt_parse", {}).get("menu", [])
    if isinstance(menu, dict):
        menu = [menu]
    for m in menu:
        if not isinstance(m, dict):
            continue
        name = m.get("nm")
        price = m.get("price")
        unit = m.get("unitprice")
        cnt = m.get("cnt")

        def _num(v):
            if v is None:
                return None
            if isinstance(v, list):
                v = v[0] if v else None
            if v is None:
                return None
            s = str(v).replace(",", "").replace(".", "") if str(v).count(".") > 1 else str(v).replace(",", "")
            s = "".join(ch for ch in s if ch.isdigit() or ch in ".-")
            try:
                return float(s)
            except ValueError:
                return None

        if isinstance(name, list):
            name = name[0] if name else None
        if name is None:
            continue
        items.append(
            {
                "item_name": str(name),
                "item_amount": _num(price) or 0.0,
                "item_rate": _num(unit),
                "item_quantity": _num(cnt),
            }
        )
    return items


def build_cord(image_dir: Path) -> None:
    from datasets import load_dataset

    image_dir.mkdir(parents=True, exist_ok=True)

    for split_name, out_name in (("train", "stage_b_cord_train"), ("validation", "stage_b_cord_eval")):
        ds = load_dataset("naver-clova-ix/cord-v2", split=split_name)
        rows = []
        for i, ex in enumerate(ds):
            gt = json.loads(ex["ground_truth"])
            items = _cord_items(gt)
            if not items:
                continue
            img_path = image_dir / f"cord_{split_name}_{i:05d}.png"
            ex["image"].convert("RGB").save(img_path)
            rows.append(
                {
                    "image": str(img_path),
                    "task": "extract",
                    "prompt": EXTRACT_PROMPT,
                    "target": json.dumps(
                        {"page_type": "Bill Detail", "bill_items": items},
                        ensure_ascii=False,
                    ),
                    "source": "cord",
                }
            )
        _write(rows, out_name)


# ── Stage C ────────────────────────────────────────────────────────────────
def build_medical() -> None:
    train_path = LABELS_DIR / "train_distilled.jsonl"
    if not train_path.exists():
        raise SystemExit(f"{train_path} missing. Run `python -m vlm.data.distill` first.")

    cls_prompt = CLASSIFY_PROMPT.format(classes="\n".join(f"- {c}" for c in PAGE_TYPES))
    rows = []
    for line in train_path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        rows.append(
            {
                "image": r["abs_path"],
                "task": "classify",
                "prompt": cls_prompt,
                "target": r["page_type"],
                "source": "medical",
                "page_id": r["page_id"],
            }
        )
        rows.append(
            {
                "image": r["abs_path"],
                "task": "extract",
                "prompt": EXTRACT_PROMPT,
                "target": json.dumps(
                    {"page_type": r["page_type"], "bill_items": r["bill_items"]},
                    ensure_ascii=False,
                ),
                "source": "medical",
                "page_id": r["page_id"],
            }
        )

    random.Random(SEED).shuffle(rows)
    _write(rows, "stage_c_medical_train")
    print(
        "\n  Note: no medical eval file is written here. The held-out medical\n"
        "  pages live in artifacts/labels/test_review.csv and must be\n"
        "  hand-verified before use. See vlm/data/distill.py."
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["rvl_cdip", "cord", "medical"])
    ap.add_argument("--per-class", type=int, default=400)
    ap.add_argument("--image-dir", type=Path, default=DATASET_DIR / "images")
    args = ap.parse_args()

    if args.stage == "rvl_cdip":
        build_rvl_cdip(args.per_class, args.image_dir)
    elif args.stage == "cord":
        build_cord(args.image_dir)
    else:
        build_medical()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
