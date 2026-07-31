"""
Central configuration for the VLM fine-tuning track.

Everything that a reviewer might want to change lives here so the training and
evaluation scripts stay free of magic numbers.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
VLM_ROOT = REPO_ROOT / "vlm"
ARTIFACT_ROOT = Path(os.getenv("VLM_ARTIFACT_ROOT", VLM_ROOT / "artifacts"))

PAGES_DIR = ARTIFACT_ROOT / "pages"            # rendered page PNGs
LABELS_DIR = ARTIFACT_ROOT / "labels"          # distilled + reviewed labels
DATASET_DIR = ARTIFACT_ROOT / "datasets"       # chat-format JSONL
ADAPTER_DIR = ARTIFACT_ROOT / "adapters"       # LoRA weights
RESULTS_DIR = ARTIFACT_ROOT / "results"        # benchmark output

for _d in (PAGES_DIR, LABELS_DIR, DATASET_DIR, ADAPTER_DIR, RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ── Label taxonomy ─────────────────────────────────────────────────────────
# Must stay in sync with app/schemas.py PageLineItems.page_type.
PAGE_TYPES = [
    "Bill Summary",
    "Bill Detail",
    "Pharmacy Bill",
    "Lab Bill",
    "Other",
]
PAGE_TYPE_TO_ID = {name: i for i, name in enumerate(PAGE_TYPES)}

# RVL-CDIP's 16 classes, in the dataset's native label order.
# We do NOT map these onto PAGE_TYPES: the taxonomies genuinely do not align.
# Stage A uses RVL-CDIP to teach the adapter the *task format* -- look at a page
# image, emit a single document-class token -- on a dataset large enough to
# actually move weights. Stage C then re-adapts the same adapter to PAGE_TYPES.
# This is transfer of task shape, not of label semantics; see vlm/README.md.
RVL_CDIP_CLASSES = [
    "letter", "form", "email", "handwritten", "advertisement",
    "scientific report", "scientific publication", "specification",
    "file folder", "news article", "budget", "invoice", "presentation",
    "questionnaire", "resume", "memo",
]


# ── Model / hardware ───────────────────────────────────────────────────────
@dataclass
class ModelConfig:
    model_id: str = os.getenv("VLM_MODEL_ID", "Qwen/Qwen2-VL-2B-Instruct")

    # Turing (T4) has no bf16 and no FlashAttention-2. Both must stay off or
    # training dies with a dtype mismatch mid-backward. Flip these on an
    # Ampere+ card (A100/4090) for a meaningful speedup.
    use_bf16: bool = os.getenv("VLM_USE_BF16", "false").lower() == "true"
    attn_implementation: str = os.getenv("VLM_ATTN", "sdpa")

    load_in_4bit: bool = os.getenv("VLM_LOAD_4BIT", "true").lower() == "true"

    # Qwen2-VL bills vision tokens as (pixels / 28 / 28) / 4 after the 2x2 merge.
    # 1024 * 28 * 28 -> ~256 visual tokens/page, which is the lowest setting at
    # which dense bill tables stayed legible in our spot checks. Drop to 768 if
    # you OOM on a 16GB T4; raise to 2048 on an A100.
    min_pixels: int = int(os.getenv("VLM_MIN_PIXELS", 256 * 28 * 28))
    max_pixels: int = int(os.getenv("VLM_MAX_PIXELS", 1024 * 28 * 28))


@dataclass
class LoraConfig_:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    # Suffixes of the Linear layers we adapt. The vision tower is excluded by
    # name at build time (see train/train_lora.py::find_target_modules) so the
    # ViT stays frozen -- it already sees documents well, and freezing it saves
    # roughly 3GB of optimizer state.
    target_suffixes: tuple = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )
    freeze_vision_tower: bool = True


# ── Pricing (verified 2026-07-31) ──────────────────────────────────────────
# Gemini 2.5 Flash list price. Override via env if these drift -- they will.
# Note: the 2.5 family is scheduled for retirement 2026-10-16, which is itself
# an argument for owning the weights.
GEMINI_INPUT_USD_PER_MTOK = float(os.getenv("GEMINI_INPUT_PRICE", "0.15"))
GEMINI_OUTPUT_USD_PER_MTOK = float(os.getenv("GEMINI_OUTPUT_PRICE", "1.25"))
GEMINI_PRICE_AS_OF = "2026-07-31"

# Rented-GPU reference points, USD/hour, on-demand.
GPU_HOURLY_USD = {
    "T4": float(os.getenv("GPU_PRICE_T4", "0.35")),
    "A10G": float(os.getenv("GPU_PRICE_A10G", "0.75")),
    "A100-40GB": float(os.getenv("GPU_PRICE_A100", "1.29")),
}

# On-device deployment (HyperVerge's SDK model) has zero marginal inference
# cost. The honest accounting is amortised engineering + distribution, which we
# report separately rather than folding into a per-page number.
ONDEVICE_MARGINAL_USD_PER_PAGE = 0.0


# ── Document grouping (leakage control) ────────────────────────────────────
# The 15 "samples" are NOT 15 independent documents. Verified by identifier
# match: train_sample_9 and train_sample_10 are the same 90-page bill
# (Bill No INT2043376, IP AMHLIP398580) sliced into contiguous page ranges --
# sample_10 is "Page 1-3 of 90", sample_9 is "Page 4-6 of 90".
#
# Splitting them across the train/test boundary leaks the exact same document,
# hospital template, patient and scanner into both sides. Anything mapped here
# is treated as ONE unit by the splitter.
#
# Only digitally-searchable duplicates could be detected automatically; run
# `python -m vlm.data.find_duplicates` and eyeball the scanned ones before
# trusting any held-out number.
DOCUMENT_GROUPS = {
    "train_sample_9.pdf": "bill_INT2043376",
    "train_sample_10.pdf": "bill_INT2043376",
}


def group_key(source_pdf: str) -> str:
    """Splittable unit for a source file. Defaults to the file itself."""
    return DOCUMENT_GROUPS.get(source_pdf, source_pdf)


# ── Reproducibility ────────────────────────────────────────────────────────
SEED = 20260731
TEST_FRACTION = 0.30
BOOTSTRAP_RESAMPLES = 10_000


@dataclass
class Paths:
    pages: Path = field(default_factory=lambda: PAGES_DIR)
    labels: Path = field(default_factory=lambda: LABELS_DIR)
    datasets: Path = field(default_factory=lambda: DATASET_DIR)
    adapters: Path = field(default_factory=lambda: ADAPTER_DIR)
    results: Path = field(default_factory=lambda: RESULTS_DIR)


MODEL = ModelConfig()
LORA = LoraConfig_()
PATHS = Paths()
