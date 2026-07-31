# Replacing the Gemini API with a LoRA-adapted Qwen2-VL-2B

MediData originally sent every bill page to Gemini 2.5 Flash. This track replaces
that call with an open-weight 2B vision-language model, fine-tuned with QLoRA on
document data, and measures what the swap actually costs in accuracy, latency,
and money.

The point is not that the small model wins. It is that you can **state the
tradeoff with numbers you produced yourself**, and that the weights are yours —
no per-page fee, no data leaving the host, and a plausible path to running
inside a mobile SDK.

---

## Status

**The code is complete and tested. No model has been trained yet.**

Every result table in this document is empty on purpose. `benchmark.py` fills
them in when you run it. Nothing here quotes an accuracy or cost figure that
hasn't been measured, and you shouldn't either until the run completes.

What *is* verified: the scoring math, the cost model, and PDF rendering all have
passing tests (see [Verification](#verification)).

---

## Why it's built this way

### The dataset was the binding constraint, not the code

The source corpus is 15 PDFs / 50 pages, unlabeled. That is not enough to
fine-tune a 2B VLM, and more importantly it is not enough to *evaluate* one.

Worse, they are not even 15 independent documents — `train_sample_9` and
`train_sample_10` are contiguous slices of the same 90-page bill. See
[CORPUS.md](CORPUS.md) for the full inspection.

The grouped document-level split lands at **32 training / 18 held-out pages**
from 10 / 4 documents. At n=18 the 95% bootstrap CI on accuracy is roughly
**±8 to ±16 points** depending on the error rate — measured, not asserted:

```
n=22, 3 errors: accuracy=0.864, 95% CI = [0.727, 1.000]   (±14 pts)
n=22, 1 error : accuracy=0.955, 95% CI = [0.864, 1.000]   (± 7 pts)
```

So at three mistakes, "86% accurate" and "73% accurate" are the same
measurement. Any interviewer who asks "what's your n?" ends the conversation
there.

And because the corpus is dominated by Bill Detail pages, **always guessing the
majority class scores roughly 60–75%**. `benchmark.py` reports that baseline
row automatically; any result at or below it has learned nothing.

Three things follow, and they shape the whole design:

1. **Training data comes mostly from public corpora.** RVL-CDIP (document
   classification) and CORD-v2 (receipt line items) supply thousands of labeled
   examples. The 50 medical pages are the *adaptation* set, not the training set.
2. **Every number ships with a confidence interval**, and the local-vs-Gemini
   comparison uses a **paired bootstrap** on the same items — the correct test
   for "is A worse than B", rather than eyeballing two overlapping CIs.
3. **The benchmark refuses to run on unverified labels.** See below.

### Teacher labels are not ground truth

The obvious shortcut is to label all 50 pages with Gemini and train on that.
That's fine for *training* — it's what distillation is. It is fatal for
*evaluation*: scoring the student against teacher output measures agreement with
Gemini, caps the student at 100% by construction, and silently inherits every
teacher error.

So `distill.py` splits first, then writes teacher labels for the training pages
and a **`test_review.csv` for you to hand-correct**. `benchmark.py` hard-fails
until you flip the `VERIFIED` flag. The difference between "agrees with Gemini
94% of the time" and "is 94% accurate" is the entire credibility of the result.

Line-item labels remain teacher-derived even after review (correcting 50 pages
of item tables by hand is hours of work). The benchmark report says so in
writing, in the `extraction_reference` field.

### The split is by document, not by page

Pages from one bill share a template, a hospital, and a scanner. Splitting by
page leaks all of that across the boundary and inflates test accuracy. Grouping
by source PDF is the only defensible option at this size — it also means the
test set is 4 documents, which is *why* the CIs are wide.

Grouping by *file* isn't sufficient either: two supplied files are slices of one
bill. `config.DOCUMENT_GROUPS` collapses known-related files into a single
splittable unit, and `python -m vlm.data.find_duplicates` re-checks. That check
only sees the 3 files with a text layer — undetected pairs among the 12 scanned
files are the biggest open risk to the held-out numbers.

### Three training stages

| stage | data | n | what it buys |
|---|---|---|---|
| A | RVL-CDIP, 16 doc classes | ~6,400 | the task *shape*: page image in, class label out |
| B | CORD-v2 receipts | ~800 | line-item extraction with a real sample size |
| C | your distilled medical pages | 28 pages → 56 examples | domain adaptation |

Stage C alone cannot teach a 2B model a new output format *and* a new domain
from 28 pages. A and B buy the format so C only has to buy the domain.

Note that RVL-CDIP's 16 classes do **not** map onto MediData's 5 page types.
Stage A is transfer of *task format*, not label semantics. Claiming otherwise
would be the kind of thing that unravels under one follow-up question.

---

## Hardware constraints (Turing / T4)

The defaults target a free Colab T4, which is Turing (sm_75):

- **fp16 only.** No bf16 units; mixing bf16 compute into an fp16 graph dies at
  the first matmul.
- **SDPA attention, not FlashAttention-2.** FA2 needs sm_80+.
- **Liger kernels are incompatible with 4-bit QLoRA.** Leave them off.
- **Vision tower frozen.** Saves ~3GB of optimizer state; the ViT already reads
  documents, what needs teaching is the output format.
- **Pixel budget capped** at `1024 × 28 × 28` ≈ 256 visual tokens/page. Drop to
  768 if you OOM; raise to 2048 on an A100.

On Ampere+, set `VLM_USE_BF16=true` and `VLM_ATTN=flash_attention_2`.

---

## Running it

Easiest path is [`notebooks/finetune_qwen2vl_colab.ipynb`](notebooks/finetune_qwen2vl_colab.ipynb).
Manually:

```bash
pip install -r requirements-vlm.txt

# 1. Rasterise the PDFs (poppler or PyMuPDF)
python -m vlm.data.render --zip attachments.zip

# 2. Teacher pass. --dry-run prints a cost estimate first.
export GOOGLE_API_KEY=...
python -m vlm.data.distill --dry-run
python -m vlm.data.distill

# 3. >>> Hand-correct vlm/artifacts/labels/test_review.csv <<<
#    then set verified=true in vlm/artifacts/labels/VERIFIED

# 4. Datasets
python -m vlm.data.build_datasets --stage rvl_cdip --per-class 400
python -m vlm.data.build_datasets --stage cord
python -m vlm.data.build_datasets --stage medical

# 5. Staged training, each warm-starting from the last
python -m vlm.train.train_lora --data stage_a_rvlcdip_train  --out stage_a               --epochs 1 --lr 1e-4
python -m vlm.train.train_lora --data stage_b_cord_train     --out stage_b --init stage_a --epochs 2 --lr 1e-4
python -m vlm.train.train_lora --data stage_c_medical_train  --out stage_c --init stage_b --epochs 6 --lr 5e-5

# 6. Benchmark
python -m vlm.eval.benchmark --backends local gemini --adapter stage_c \
    --deployment T4 --training-gpu-hours 3.0
```

Rough T4 budget: stage A 45–70 min, stage B 30–60 min, stage C a few minutes.

### Serving

```bash
EXTRACTOR_BACKEND=local VLM_ADAPTER=stage_c uvicorn app.main:app
```

`app/local_backend.py` mirrors `BillExtractor`'s interface, so routes are
unchanged and `/health` reports which backend is live. It falls back to Gemini
with a loud log line if the adapter or GPU is missing.

---

## Results

Run the benchmark, then paste the generated
`artifacts/results/benchmark_*.md` here.

### Accuracy — page classification

| backend | accuracy | 95% CI | macro-F1 | 95% CI |
|---|---|---|---|---|
| Majority-class baseline | — | — | — | — |
| Gemini 2.5 Flash | — | — | — | — |
| Qwen2-VL-2B base (no adapter) | — | — | — | — |
| Qwen2-VL-2B + LoRA (stage C) | — | — | — | — |

Report printed vs handwritten pages separately — [CORPUS.md §5](CORPUS.md)
predicts the gap is concentrated almost entirely in the handwritten pharmacy
memos, which is a more useful finding than a single averaged number.

The base-model row is the ablation that matters: if stage C isn't clearly above
it, the fine-tune did nothing and the honest report says so.

### Accuracy — line-item extraction

| backend | item micro-F1 | 95% CI | amount MAE | grand-total error |
|---|---|---|---|---|
| Qwen2-VL-2B + LoRA | — | — | — | — |

Reference is teacher output, not ground truth. Report it as agreement.

### Paired comparison

| metric | value |
|---|---|
| Δ accuracy (local − Gemini) | — |
| 95% CI | — |
| p | — |

**If the CI straddles zero, there is no measured gap.** Say "indistinguishable
at n=22", not a point estimate.

### Latency

| backend | p50 | p90 | p95 |
|---|---|---|---|
| Gemini (incl. network) | — | — | — |
| Local (T4, 4-bit) | — | — | — |

Model load (~20–40s) is excluded and warmup calls are discarded — the first
generate is several times slower than steady state.

### Cost

| | value |
|---|---|
| Gemini, $/1000 pages | — |
| Local marginal, $/1000 pages | — |
| Training, one-off | — |
| **Breakeven volume** | — |

Gemini 2.5 Flash list price as of 2026-07-31: **$0.15/M input, $1.25/M output**.
Batch API is 50% cheaper. Note the 2.5 family retires 2026-10-16 — which is
itself an argument for owning weights.

---

## How to talk about the results

The tempting line is *"6% accuracy drop, 4× cheaper, runs on-device."* Two of
those three are claims that fall apart under questioning:

- **"6% drop"** — not measurable at n=22. The CI is wider than the effect.
  Say: *"indistinguishable from the API on a 22-page held-out set — the
  confidence interval is ±14 points, so the honest read is that I can't
  detect a difference at this sample size, and here's the public-data eval
  where n is large enough to say something."*

- **"4× cheaper"** — hides a volume assumption. Local inference has a fixed
  training cost and a marginal cost that only beats the API above some volume;
  below it the API is simply cheaper. `cost_model.py` computes the crossover.
  Say: *"cheaper above ~N pages on a T4; below that the API wins, and here's
  the breakeven."*

- **"runs on-device"** — this one holds, and it's the strongest of the three.
  A 2B model at 4-bit is ~1.5GB. The adapter alone is ~40MB. No PHI leaves the
  device, no per-page fee, works offline. For a company whose SDK does
  on-device processing, that's the argument — not the cost multiplier.

The version that survives follow-up: *"I distilled the hosted API into a 2B VLM
I fine-tuned myself, and I built the evaluation honestly enough to tell you
exactly where it's worse and at what volume it pays for itself."*

---

## Layout

```
vlm/
├── config.py                 # all knobs: paths, labels, pricing, hardware
├── data/
│   ├── render.py             # PDF -> page PNGs (poppler, PyMuPDF fallback)
│   ├── distill.py            # Gemini teacher -> labels + review CSV
│   └── build_datasets.py     # RVL-CDIP / CORD / medical -> chat JSONL
├── train/train_lora.py       # QLoRA, T4-safe, prompt-masked loss
├── eval/
│   ├── metrics.py            # stdlib-only scoring + bootstrap CIs
│   ├── cost_model.py         # API vs local, with breakeven
│   └── benchmark.py          # head-to-head runner + report
├── serve/local_vlm.py        # base + adapter inference
└── notebooks/                # Colab T4 walkthrough
```

`app/local_backend.py` wires the local model into the existing FastAPI service.

---

## Verification

`metrics.py` and `cost_model.py` are deliberately stdlib-only so scoring can be
tested on any machine, with no CUDA and no ML stack:

```bash
python -m compileall -q vlm app      # all modules compile
python tests/test_metrics.py         # 24 assertions on the scoring math
python tests/test_cost.py            # 12 assertions on the cost math
```

Also verified end-to-end on the real corpus: all 50 pages rasterise, the
manifest is written, and the document-level split is deterministic with zero
document overlap between train and test.

Covered: perfect and known-accuracy classifiers, degenerate CIs, macro-F1 with
absent classes, empirical CI width at small n, paired bootstrap under identical and
strictly-dominant systems, item matching (name variants match, wrong amounts
don't), percentile interpolation against hand-computed values, and every cost
path including the no-breakeven case where the API wins at all volumes.

---

## Known gaps

- **Extraction labels are teacher-derived.** Item-level F1 measures agreement
  with Gemini. Hand-correcting the item tables would fix this and is the single
  highest-value few hours available.
- **One seed.** LoRA at this data scale has real run-to-run variance; 3–5 seeds
  with the spread reported would be more honest than a single number.
- **RVL-CDIP and CORD are not medical bills.** Stage A/B transfer is an
  assumption the benchmark does not isolate. A stage-C-only ablation would
  measure it.
- **No quantized on-device measurement.** The on-device claim is currently
  architectural, not measured. Exporting to GGUF/ONNX and timing it on a phone
  would convert the strongest argument from plausible to demonstrated.
